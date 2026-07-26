import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "comparison_claim_matrix.csv": [
        "comparison_id", "treatment", "control", "scientific_question",
        "changed_factors", "same_split", "same_cases", "same_backbone",
        "same_capacity", "same_optimizer", "same_checkpoint_rule", "same_metric",
        "seed_control", "fold_coverage", "source_faithful", "validity_level",
        "allowed_claim", "forbidden_claim", "blocker", "evidence_path",
    ],
    "pipeline_correctness_checklist.csv": [
        "check_id", "requirement", "audit_method", "status", "finding",
        "impact_on_historical_results", "file_class_function_line",
        "minimal_fix_or_stage3_requirement", "evidence_path",
    ],
    "metric_definition_audit.csv": [
        "metric_id", "definition", "aggregation", "empty_gt_empty_pred",
        "gt_present_pred_empty", "gt_absent_pred_present", "unit",
        "historical_use", "stage3_role", "comparability_and_audit_finding",
        "evidence",
    ],
    "seed_and_split_audit.csv": [
        "experiment_family", "folds", "cases_per_fold", "split_source",
        "split_hash", "same_case_split_verified", "model_seed_status",
        "data_seed_status", "augmentation_worker_status", "determinism_status",
        "paired_interpretation", "conclusion", "evidence",
    ],
}

for name, expected_fields in EXPECTED.items():
    with (ROOT / name).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == expected_fields, (name, reader.fieldnames)
    assert rows and all(None not in row for row in rows), name
    print(name, len(rows), "rows OK")

for name in [
    "audit_manifest.json",
    "tests/current_pipeline_synthetic_results.json",
    "tests/real_data_and_summary_audit.json",
    "tests/existing_verifier_runs.json",
]:
    json.loads((ROOT / name).read_text(encoding="utf-8"))
    print(name, "JSON OK")

for name in [
    "stage3_preregistered_protocol.md",
    "stage3_required_tests.md",
    "COMPARISON_VALIDITY_AUDIT.md",
]:
    text = (ROOT / name).read_text(encoding="utf-8")
    assert len(text) > 1000, (name, len(text))
    print(name, len(text), "characters OK")

report = (ROOT / "COMPARISON_VALIDITY_AUDIT.md").read_text(encoding="utf-8")
assert report.count("STAGE3_BLOCKED") == 1
assert "STAGE3_READY" not in report
assert "STAGE3_READY_WITH_FIXES" not in report
print("final status uniqueness OK")
