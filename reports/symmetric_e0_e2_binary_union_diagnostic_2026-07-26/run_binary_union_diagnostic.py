from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree
from scipy.stats import spearmanr


SEEDS=(3407,1234,5678)
ARMS=("E0","E2")
MASK_TYPES=("hard_union","soft_union_p0p5")
BOOTSTRAP_SEED=20260726
N_BOOTSTRAP=10_000
GIT_COMMIT="0c660e0"


def dice(reference:np.ndarray,prediction:np.ndarray)->float:
    nr=int(np.count_nonzero(reference));npred=int(np.count_nonzero(prediction))
    if nr+npred==0:return math.nan
    tp=int(np.count_nonzero(reference & prediction))
    return float(2*tp/(nr+npred))


def hd95_mm(reference:np.ndarray,prediction:np.ndarray,spacing:tuple[float,float,float])->float:
    if not np.any(reference) or not np.any(prediction):return math.nan
    structure=np.ones((3,3,3),dtype=bool)
    ref_surface=reference ^ binary_erosion(reference,structure=structure,border_value=0)
    pred_surface=prediction ^ binary_erosion(prediction,structure=structure,border_value=0)
    ref_points=np.argwhere(ref_surface)*np.asarray(spacing,dtype=np.float64)
    pred_points=np.argwhere(pred_surface)*np.asarray(spacing,dtype=np.float64)
    pred_to_ref=cKDTree(ref_points).query(pred_points,k=1,workers=-1)[0]
    ref_to_pred=cKDTree(pred_points).query(ref_points,k=1,workers=-1)[0]
    distances=np.concatenate((pred_to_ref,ref_to_pred))
    return float(np.percentile(distances,95))


def finite_mean(values)->float:
    vals=[float(v) for v in values if math.isfinite(float(v))]
    return statistics.fmean(vals) if vals else math.nan


def sample_sd(values)->float:
    vals=[float(v) for v in values if math.isfinite(float(v))]
    return statistics.stdev(vals) if len(vals)>1 else math.nan


def lesion_group(volume_ml:float)->str:
    if volume_ml<1:return "small_lt1ml"
    if volume_ml<10:return "medium_1to10ml"
    return "large_ge10ml"


def write_csv(path:Path,rows:list[dict])->None:
    fields=[];seen=set()
    for row in rows:
        for key in row:
            if key not in seen:fields.append(key);seen.add(key)
    with path.open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(np.floating,float)):
        v=float(value);return v if math.isfinite(v) else None
    return value


def write_json(path:Path,value)->None:
    path.write_text(json.dumps(json_safe(value),indent=2,sort_keys=True)+"\n",encoding="utf-8")


def result_paths(drive_root:Path,multiseed_root:Path)->dict[tuple[str,int],Path]:
    names={
      ("E0",3407):"nnUNetTrainer_25D_SymmetricE0Control__nnUNetPlans__2d",
      ("E2",3407):"nnUNetTrainer_25D_SymmetricE2ReliabilityGate__nnUNetPlans__2d",
      ("E0",1234):"nnUNetTrainer_25D_SymmetricE0ControlSeed1234__nnUNetPlans__2d",
      ("E2",1234):"nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed1234__nnUNetPlans__2d",
      ("E0",5678):"nnUNetTrainer_25D_SymmetricE0ControlSeed5678__nnUNetPlans__2d",
      ("E2",5678):"nnUNetTrainer_25D_SymmetricE2ReliabilityGateSeed5678__nnUNetPlans__2d"}
    return {key:(drive_root if key[1]==3407 else multiseed_root)/name/"fold_0"/"validation" for key,name in names.items()}


def verify_sets(paths:dict,gt_root:Path)->list[str]:
    reference=None
    for key,val in paths.items():
        ids=sorted(p.name[:-7] for p in val.glob("case_*.nii.gz"))
        if len(ids)!=39 or len(ids)!=len(set(ids)):raise RuntimeError(f"invalid case set {key}: {len(ids)}")
        if reference is None:reference=ids
        if ids!=reference:raise RuntimeError(f"case set mismatch {key}")
        for cid in ids:
            for suffix in (".nii.gz",".npz",".pkl"):
                if not Path(str(val/cid)+suffix).is_file():raise FileNotFoundError(f"{key} {cid}{suffix}")
            if not (gt_root/f"{cid}.nii.gz").is_file():raise FileNotFoundError(gt_root/f"{cid}.nii.gz")
    return reference or []


def case_metrics(gt:np.ndarray,pred:np.ndarray,spacing:tuple[float,float,float],voxel_ml:float)->dict:
    tp=int(np.count_nonzero(gt & pred));fp=int(np.count_nonzero((~gt)&pred));fn=int(np.count_nonzero(gt&(~pred)))
    nr=tp+fn;npred=tp+fp
    return {"dice":dice(gt,pred),"hd95_mm":hd95_mm(gt,pred,spacing),"hd95_defined":bool(nr and npred),
            "complete_miss":bool(nr and not npred),"tp_voxels":tp,"fp_voxels":fp,"fn_voxels":fn,
            "fp_volume_ml":fp*voxel_ml,"fn_volume_ml":fn*voxel_ml,
            "precision":tp/npred if npred else math.nan,"recall":tp/nr if nr else math.nan,
            "pred_volume_ml":npred*voxel_ml,"gt_volume_ml":nr*voxel_ml}


def compute(paths:dict,gt_root:Path,case_ids:list[str])->tuple[list[dict],dict]:
    rows=[];soft_audit={}
    for seed in SEEDS:
        for cid in case_ids:
            gt_img=nib.load(str(gt_root/f"{cid}.nii.gz"));gt_mc=np.asanyarray(gt_img.dataobj)
            if not set(int(v) for v in np.unique(gt_mc))<=set(range(6)):raise RuntimeError(f"illegal GT label {cid}")
            gt=gt_mc>0;spacing=tuple(float(v) for v in gt_img.header.get_zooms()[:3])
            voxel_ml=abs(float(np.linalg.det(gt_img.affine[:3,:3])))/1000
            zoom_ml=float(np.prod(spacing))/1000
            if not math.isclose(voxel_ml,zoom_ml,rel_tol=1e-5,abs_tol=1e-8):raise RuntimeError(f"voxel volume mismatch {cid}")
            total_ml=int(np.count_nonzero(gt))*voxel_ml;group=lesion_group(total_ml)
            for arm in ARMS:
                val=paths[(arm,seed)];base=val/cid
                pred_img=nib.load(str(base)+".nii.gz");pred_mc=np.asanyarray(pred_img.dataobj)
                if pred_mc.shape!=gt_mc.shape or not np.allclose(pred_img.affine,gt_img.affine,atol=1e-5,rtol=0):
                    raise RuntimeError(f"geometry mismatch {arm} {seed} {cid}")
                if not set(int(v) for v in np.unique(pred_mc))<=set(range(6)):raise RuntimeError(f"illegal prediction {arm} {seed} {cid}")
                with Path(str(base)+".pkl").open("rb") as handle:props=pickle.load(handle)
                with np.load(str(base)+".npz") as payload:prob=payload["probabilities"]
                shape_before=tuple(int(v) for v in props["shape_before_cropping"])
                shape_crop=tuple(int(v) for v in props["shape_after_cropping_and_before_resampling"])
                bbox=props["bbox_used_for_cropping"]
                full_bbox=all(int(bounds[0])==0 and int(bounds[1])==shape_before[i] for i,bounds in enumerate(bbox))
                if prob.shape!=(6,*shape_crop) or shape_crop!=shape_before or not full_bbox:
                    raise RuntimeError(f"unsafe probability restoration {arm} {seed} {cid}")
                if not np.array_equal(np.argmax(prob,axis=0),pred_mc.transpose(2,1,0)):
                    raise RuntimeError(f"probability argmax mismatch {arm} {seed} {cid}")
                hard_union=pred_mc>0
                soft_union=(1.0-prob[0]).transpose(2,1,0)>=0.5
                soft_audit[f"{arm}_{seed}_{cid}"]={"safe":True,"hard_soft_voxel_disagreement":int(np.count_nonzero(hard_union!=soft_union))}
                binary_tp=int(np.count_nonzero(gt&hard_union))
                correct_subtype=int(np.count_nonzero((gt_mc==pred_mc)&(gt_mc>0)))
                wrong_subtype=int(np.count_nonzero((gt_mc>0)&(pred_mc>0)&(gt_mc!=pred_mc)))
                for mask_type,mask in (("hard_union",hard_union),("soft_union_p0p5",soft_union)):
                    row={"case_id":cid,"seed":seed,"model":arm,"mask_type":mask_type,"lesion_group":group,
                         "spacing_x_mm":spacing[0],"spacing_y_mm":spacing[1],"spacing_z_mm":spacing[2],
                         "voxel_volume_ml":voxel_ml,"subtype_correct_foreground_voxels":correct_subtype,
                         "wrong_subtype_foreground_voxels":wrong_subtype,
                         "wrong_subtype_fraction_of_hard_binary_tp":wrong_subtype/binary_tp if binary_tp else math.nan}
                    row.update(case_metrics(gt,mask,spacing,voxel_ml));rows.append(row)
    return rows,soft_audit


def aggregate(rows:list[dict])->tuple[list[dict],list[dict]]:
    aggregate_rows=[];group_rows=[]
    for mask_type in MASK_TYPES:
      for seed in SEEDS:
       for arm in ARMS:
        subset=[r for r in rows if r["mask_type"]==mask_type and r["seed"]==seed and r["model"]==arm]
        aggregate_rows.append({"mask_type":mask_type,"seed":seed,"model":arm,"n_cases":len(subset),
          "mean_dice":finite_mean(r["dice"] for r in subset),"sample_sd_dice":sample_sd(r["dice"] for r in subset),
          "median_dice":float(np.median([r["dice"] for r in subset])),"minimum_dice":min(r["dice"] for r in subset),
          "mean_hd95_mm_defined":finite_mean(r["hd95_mm"] for r in subset),"hd95_defined_cases":sum(r["hd95_defined"] for r in subset),
          "complete_miss_cases":sum(r["complete_miss"] for r in subset),"mean_fp_volume_ml":finite_mean(r["fp_volume_ml"] for r in subset),
          "mean_fn_volume_ml":finite_mean(r["fn_volume_ml"] for r in subset),"mean_precision":finite_mean(r["precision"] for r in subset),
          "mean_recall":finite_mean(r["recall"] for r in subset)})
        for group in ("small_lt1ml","medium_1to10ml","large_ge10ml"):
          selected=[r for r in subset if r["lesion_group"]==group]
          group_rows.append({"mask_type":mask_type,"seed":seed,"model":arm,"lesion_group":group,"n_cases":len(selected),
            "mean_dice":finite_mean(r["dice"] for r in selected),"sample_sd_dice":sample_sd(r["dice"] for r in selected),
            "mean_hd95_mm_defined":finite_mean(r["hd95_mm"] for r in selected),"hd95_defined_cases":sum(r["hd95_defined"] for r in selected),
            "complete_miss_cases":sum(r["complete_miss"] for r in selected),"mean_fp_volume_ml":finite_mean(r["fp_volume_ml"] for r in selected),
            "mean_recall":finite_mean(r["recall"] for r in selected),"notes":"pre-existing whole-case volume thresholds; descriptive"})
    return aggregate_rows,group_rows


def bootstrap(rows:list[dict])->dict:
    rng=np.random.default_rng(BOOTSTRAP_SEED);case_ids=sorted({r["case_id"] for r in rows})
    by_key={(r["mask_type"],r["seed"],r["model"],r["case_id"]):r for r in rows}
    output={}
    for mask_type in MASK_TYPES:
      per_seed={}
      for seed in SEEDS:
        observed=finite_mean(by_key[(mask_type,seed,"E2",c)]["dice"]-by_key[(mask_type,seed,"E0",c)]["dice"] for c in case_ids)
        values=[]
        for _ in range(N_BOOTSTRAP):
          sample=[case_ids[i] for i in rng.integers(0,len(case_ids),len(case_ids))]
          values.append(finite_mean(by_key[(mask_type,seed,"E2",c)]["dice"]-by_key[(mask_type,seed,"E0",c)]["dice"] for c in sample))
        per_seed[str(seed)]={"observed_mean_paired_dice_delta_e2_minus_e0":observed,"bootstrap_mean":finite_mean(values),
          "bootstrap_sample_sd":sample_sd(values),"percentile_95_ci":[float(v) for v in np.percentile(values,[2.5,97.5])],"valid_replicates":len(values)}
      joint_observed=finite_mean(per_seed[str(s)]["observed_mean_paired_dice_delta_e2_minus_e0"] for s in SEEDS)
      joint=[]
      for _ in range(N_BOOTSTRAP):
        sample=[case_ids[i] for i in rng.integers(0,len(case_ids),len(case_ids))]
        joint.append(finite_mean(finite_mean(by_key[(mask_type,s,"E2",c)]["dice"]-by_key[(mask_type,s,"E0",c)]["dice"] for c in sample) for s in SEEDS))
      output[mask_type]={"per_seed":per_seed,"three_seed_average":{"observed":joint_observed,"bootstrap_mean":finite_mean(joint),
        "bootstrap_sample_sd":sample_sd(joint),"percentile_95_ci":[float(v) for v in np.percentile(joint,[2.5,97.5])],"valid_replicates":len(joint)}}
    return {"rng_seed":BOOTSTRAP_SEED,"n_replicates":N_BOOTSTRAP,"cluster":"patient; all model seeds retained jointly for three-seed statistic",
            "estimand":"paired case-level binary Dice E2-minus-E0","results":output,
            "limitations":["39 fold-0 validation patients","no fold uncertainty","model seeds are not independent cohorts","post-hoc collapse of multiclass models"]}


def stability(rows:list[dict])->dict:
    result={}
    for mask_type in MASK_TYPES:
      for arm in ARMS:
        means=[]
        for seed in SEEDS:
          subset=[r for r in rows if r["mask_type"]==mask_type and r["model"]==arm and r["seed"]==seed]
          means.append(finite_mean(r["dice"] for r in subset))
        pairs=[]
        for i,s1 in enumerate(SEEDS):
          for s2 in SEEDS[i+1:]:
            x=[next(r["dice"] for r in rows if r["mask_type"]==mask_type and r["model"]==arm and r["seed"]==s1 and r["case_id"]==c) for c in sorted({r["case_id"] for r in rows})]
            y=[next(r["dice"] for r in rows if r["mask_type"]==mask_type and r["model"]==arm and r["seed"]==s2 and r["case_id"]==c) for c in sorted({r["case_id"] for r in rows})]
            pairs.append(float(spearmanr(x,y).statistic))
        result[f"{mask_type}_{arm}"]={"per_seed_mean_dice":dict(zip(map(str,SEEDS),means)),"mean_across_seed_means":finite_mean(means),
          "sample_sd_across_seed_means":sample_sd(means),"pairwise_case_dice_spearman":pairs,"median_pairwise_spearman":float(np.median(pairs))}
    return result


def diagnostic(rows:list[dict],aggregate_rows:list[dict],bootstrap_payload:dict,stability_payload:dict)->dict:
    hard=[r for r in aggregate_rows if r["mask_type"]=="hard_union"]
    model_summary={arm:{"per_seed_mean_dice":{str(s):next(r["mean_dice"] for r in hard if r["model"]==arm and r["seed"]==s) for s in SEEDS},
                        "mean_of_seed_means":finite_mean(r["mean_dice"] for r in hard if r["model"]==arm),
                        "sample_sd_of_seed_means":sample_sd(r["mean_dice"] for r in hard if r["model"]==arm)} for arm in ARMS}
    small={arm:{str(s):finite_mean(r["dice"] for r in rows if r["mask_type"]=="hard_union" and r["model"]==arm and r["seed"]==s and r["lesion_group"]=="small_lt1ml") for s in SEEDS} for arm in ARMS}
    wrong_fraction={arm:finite_mean(r["wrong_subtype_fraction_of_hard_binary_tp"] for r in rows if r["mask_type"]=="hard_union" and r["model"]==arm) for arm in ARMS}
    joint=bootstrap_payload["results"]["hard_union"]["three_seed_average"]
    return {"analysis_type":"post-hoc binary union diagnostic of multiclass models; not binary-trained performance",
      "model_summary":model_summary,"small_lesion_mean_dice":small,"mean_wrong_subtype_fraction_among_hard_binary_true_positive_voxels":wrong_fraction,
      "e2_minus_e0_three_seed_paired_delta":joint,"stability":stability_payload,
      "judgment":"Use these results to decide whether foreground localization is already credible. They cannot establish binary-training performance or 2.5D benefit versus center-only 2D.",
      "required_next_comparison_if_binary_is_pursued":"binary-trained center-only 2D versus binary-trained simple symmetric 3-slice 2.5D under identical protocol"}


def report_text(diag:dict,aggregates:list[dict],groups:list[dict],bootstrap_payload:dict)->str:
    lines=["# Post-hoc binary-union diagnostic of E0/E2 multiclass predictions","",
      "**This is not a binary-trained model result.** It measures what the existing multiclass models become after all five hemorrhage labels are treated as one foreground class.","",
      "## Primary hard-union results","","| Seed | Model | Mean Dice | SD | Mean HD95 (mm) | FP (mL) | FN (mL) | Recall | Complete misses |","|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for seed in SEEDS:
      for arm in ARMS:
        r=next(x for x in aggregates if x["mask_type"]=="hard_union" and x["seed"]==seed and x["model"]==arm)
        lines.append(f"| {seed} | {arm} | {r['mean_dice']:.4f} | {r['sample_sd_dice']:.4f} | {r['mean_hd95_mm_defined']:.2f} | {r['mean_fp_volume_ml']:.3f} | {r['mean_fn_volume_ml']:.3f} | {r['mean_recall']:.4f} | {r['complete_miss_cases']} |")
    lines += ["","## Fixed-threshold soft-union results","","`P(ICH)=1-P(background)` with a pre-specified threshold of 0.5; no validation threshold search was performed.","","| Seed | Model | Mean Dice | Mean HD95 (mm) | FP (mL) | Recall |","|---:|---|---:|---:|---:|---:|"]
    for seed in SEEDS:
      for arm in ARMS:
        r=next(x for x in aggregates if x["mask_type"]=="soft_union_p0p5" and x["seed"]==seed and x["model"]==arm)
        lines.append(f"| {seed} | {arm} | {r['mean_dice']:.4f} | {r['mean_hd95_mm_defined']:.2f} | {r['mean_fp_volume_ml']:.3f} | {r['mean_recall']:.4f} |")
    lines += ["","## Small-lesion diagnostic","","The pre-existing whole-case thresholds are retained. Fold 0 has only four `<1 mL` patients, so these values are descriptive.","","| Seed | Model | n | Mean hard-union Dice | Complete misses |","|---:|---|---:|---:|---:|"]
    for seed in SEEDS:
      for arm in ARMS:
        r=next(x for x in groups if x["mask_type"]=="hard_union" and x["seed"]==seed and x["model"]==arm and x["lesion_group"]=="small_lt1ml")
        lines.append(f"| {seed} | {arm} | {r['n_cases']} | {r['mean_dice']:.4f} | {r['complete_miss_cases']} |")
    joint=bootstrap_payload["results"]["hard_union"]["three_seed_average"]
    lines += ["","## Paired E2-versus-E0 result",f"The three-seed average paired hard-union Dice delta is {joint['observed']:+.5f}; patient-bootstrap 95% CI [{joint['percentile_95_ci'][0]:+.5f}, {joint['percentile_95_ci'][1]:+.5f}]. This evaluates E2 versus E0, not 2.5D versus 2D.","",
      "## Subtype-confusion recovery",f"Among hard-union true-positive foreground voxels, the mean fraction assigned to the wrong hemorrhage subtype is {diag['mean_wrong_subtype_fraction_among_hard_binary_true_positive_voxels']['E0']:.1%} for E0 and {diag['mean_wrong_subtype_fraction_among_hard_binary_true_positive_voxels']['E2']:.1%} for E2. These voxels become correct only after subtype labels are collapsed.","",
      "## Interpretation","If hard-union Dice is strong and reproducible while multiclass subtype performance remains weak, the principal bottleneck is subtype separation rather than foreground localization. If hard-union remains weak—especially for `<1 mL` cases—the binary route does not automatically solve the localization problem.","",
      "The result cannot be reported as binary-model performance, cannot show what binary loss would learn, and cannot establish that 2.5D helps because both E0 and E2 are three-slice models.","",
      "## Decision for the next experiment","A binary training experiment is justified only as a new controlled question: binary-trained center-only 2D versus binary-trained simple symmetric 3-slice 2.5D, followed by multi-fold confirmation if the fold-0 seed effect is stable.",""]
    return "\n".join(lines)


def checksums(output:Path)->None:
    lines=[]
    for p in sorted((p for p in output.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"),key=lambda p:p.name):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (output/"SHA256SUMS.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")


def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True);parser.add_argument("--drive-root",type=Path,required=True)
    parser.add_argument("--multiseed-root",type=Path,required=True);parser.add_argument("--gt-root",type=Path,required=True);parser.add_argument("--test-file",type=Path,required=True)
    args=parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):raise FileExistsError(f"refusing overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=result_paths(args.drive_root,args.multiseed_root);case_ids=verify_sets(paths,args.gt_root)
    rows,soft_audit=compute(paths,args.gt_root,case_ids);aggregates,groups=aggregate(rows);boot=bootstrap(rows);stable=stability(rows);diag=diagnostic(rows,aggregates,boot,stable)
    write_csv(args.output_dir/"binary_union_case_metrics.csv",rows);write_csv(args.output_dir/"binary_union_aggregate.csv",aggregates)
    write_csv(args.output_dir/"binary_union_small_lesion.csv",groups);write_json(args.output_dir/"binary_union_bootstrap.json",boot)
    write_json(args.output_dir/"binary_union_diagnostic.json",diag)
    config={"date":"2026-07-26","git_commit":GIT_COMMIT,"fold":0,"model_seeds":SEEDS,"case_count":39,
      "hard_union":"prediction label >0","soft_union":"1-P(background) >=0.5; fixed threshold; no tuning",
      "dice":"2TP/(2TP+FP+FN)","hd95":"symmetric 95th percentile surface distance in physical mm; undefined for empty prediction",
      "voxel_volume_ml":"abs(det(affine[:3,:3]))/1000","small_lesion_groups":["<1mL","1-10mL",">=10mL"],
      "bootstrap_seed":BOOTSTRAP_SEED,"bootstrap_replicates":N_BOOTSTRAP}
    manifest={"case_ids":case_ids,"result_paths":{f"{a}_{s}":str(p) for (a,s),p in paths.items()},"gt_root":str(args.gt_root),
      "npz_mapping_verified_for_all_234_predictions":len(soft_audit)==234 and all(v["safe"] for v in soft_audit.values()),
      "generated_files":["binary_union_case_metrics.csv","binary_union_aggregate.csv","binary_union_small_lesion.csv","binary_union_bootstrap.json",
      "binary_union_diagnostic.json","analysis_config.json","analysis_manifest.json","BINARY_UNION_REPORT.md","run_binary_union_diagnostic.py","test_binary_union_diagnostic.py","SHA256SUMS.txt"]}
    write_json(args.output_dir/"analysis_config.json",config);write_json(args.output_dir/"analysis_manifest.json",manifest)
    (args.output_dir/"BINARY_UNION_REPORT.md").write_text(report_text(diag,aggregates,groups,boot),encoding="utf-8")
    shutil.copy2(Path(__file__),args.output_dir/"run_binary_union_diagnostic.py");shutil.copy2(args.test_file,args.output_dir/"test_binary_union_diagnostic.py");checksums(args.output_dir)
    print(json.dumps(json_safe({"output":str(args.output_dir),"diagnostic":diag}),indent=2))


if __name__=="__main__":main()
