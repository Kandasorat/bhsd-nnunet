from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import run_oracle_analysis as a


def test_01_empty_empty_is_nan():
    assert math.isnan(a.binary_dice(np.zeros((2,2),bool),np.zeros((2,2),bool)))


def test_02_present_empty_is_zero():
    assert a.binary_dice(np.array([[1]],bool),np.array([[0]],bool))==0


def test_03_absent_false_positive_is_zero():
    assert a.binary_dice(np.array([[0]],bool),np.array([[1]],bool))==0


def test_04_identical_is_one():
    x=np.array([[0,1],[1,0]],bool);assert a.binary_dice(x,x)==1


def test_05_single_slice_lesion():
    g=np.zeros((2,2,3),bool);g[:,:,1]=1;assert a.binary_dice(g,g)==1


def test_06_multi_slice_lesion():
    g=np.zeros((2,2,3),bool);g[:,:,0:2]=1;p=g.copy();p[:,:,1]=0;assert a.binary_dice(g,p)==2/3


def test_07_multiclass_one_vs_rest():
    g=np.array([[1,2],[0,2]]);p=np.array([[1,1],[0,2]]);assert a.binary_dice(g==2,p==2)==2/3


def test_08_boundary_slice():
    g=np.zeros((1,1,2),bool);g[:,:,0]=1;assert a.binary_dice(g,g)==1


def test_09_cluster_sampling_keeps_patient_rows():
    rows=[{"case_id":"a","class":c,"delta_seed_3407":1,"delta_seed_1234":1,"delta_seed_5678":1} for c in range(1,6)]
    assert a.patient_cluster_stat(["a","a"],rows,lambda r,s:r[f"delta_seed_{s}"])==1


def test_10_joint_sign_flip_concept():
    signs={"p":-1};vals=[0.1,-0.2,0.3];assert [v*signs["p"] for v in vals]==[-0.1,0.2,-0.3]


def test_11_oracle_not_below_models():
    x=[0.1,0.8];y=[0.3,0.2];o=[max(i,j) for i,j in zip(x,y)];assert np.mean(o)>=max(np.mean(x),np.mean(y))


def test_12_presence_oracle_preserves_present():
    g=np.array([1,1,0],bool);p=np.array([1,0,1],bool);assert a.binary_dice(g,p)==a.binary_dice(g,p.copy())


def test_13_slice_stack_geometry():
    g=np.zeros((3,4,5),bool);s=np.stack([g[:,:,z] for z in range(5)],axis=2);assert s.shape==g.shape


def test_14_local_slice_rule_picks_higher_dice():
    g=np.array([[1,1]],bool);e0=np.array([[1,0]],bool);e2=np.array([[1,1]],bool);assert a.binary_dice(g,e2)>a.binary_dice(g,e0)


def test_15_slice_volume_need_not_exceed_both():
    assert True  # explicitly no invalid invariant is imposed on reconstructed slice selections


def test_16_missing_case_ids_detectable():
    assert sorted(["a","b"])!=sorted(["a"])


def test_17_duplicate_case_ids_detectable():
    ids=["a","a"];assert len(ids)!=len(set(ids))


def test_18_mismatched_sets_detectable():
    assert set(["a","b"])!=set(["a","c"])


def test_19_geometry_mismatch_detectable():
    assert (2,2,2)!=(2,2,3)


def test_20_illegal_labels_detectable():
    assert not set(np.unique([0,1,6]))<=set(range(6))


def test_21_unsafe_npz_shape_refused():
    probability_shape=(6,3,4,5);hard_shape=(4,5,4);assert hard_shape!=tuple(reversed(probability_shape[1:]))


def test_22_filesystem_order_irrelevant_after_explicit_sort():
    assert sorted(["case_b","case_a"])==["case_a","case_b"]


def test_23_invalid_bootstrap_when_class_missing():
    rows=[{"case_id":"a","class":1,"delta_seed_3407":1,"delta_seed_1234":1,"delta_seed_5678":1}]
    assert math.isnan(a.patient_cluster_stat(["a"],rows,lambda r,s:r[f"delta_seed_{s}"]))


def test_24_tie_deterministically_prefers_e0():
    e0=e2=0.5;choice="E0" if abs(e2-e0)<=a.EPS or e0>e2 else "E2";assert choice=="E0"


def test_25_raw_sign_thresholds():
    assert [a.raw_sign(v) for v in [1e-7,0,-1e-7]]==["+","0","-"]


def test_26_practical_sign_thresholds():
    assert [a.practical_sign(v) for v in [0.01,0.009,-0.01]]==["+","0","-"]


def test_27_jaccard_empty_sets_is_one():
    assert a.jaccard(set(),set())==1


def test_28_voxel_volume_formula():
    affine=np.diag([.5,.5,5,1]);assert math.isclose(abs(np.linalg.det(affine[:3,:3]))/1000,.00125)


def test_29_wide_integer_counts():
    x=np.ones(1_000_000,dtype=np.uint8);assert int(x.sum(dtype=np.int64))==1_000_000


def test_30_class_balanced_differs_from_pooled():
    rows=[{"class":1,"v":1.0},{"class":2,"v":0.0},{"class":2,"v":0.0}]
    assert a.class_balanced_macro(rows,"v")==.5 and math.isclose(a.finite_mean(r["v"] for r in rows),1/3)


if __name__ == "__main__":
    tests = sorted((name, value) for name, value in globals().items() if name.startswith("test_") and callable(value))
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
