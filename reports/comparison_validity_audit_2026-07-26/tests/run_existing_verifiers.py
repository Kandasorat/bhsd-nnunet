from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
SCRIPTS = (
    "verify_attention_screen.py",
    "verify_spectral_slice_fusion.py",
    "verify_symmetric_reliability_fusion.py",
    "verify_compute_cost.py",
    "verify_symmetric_multiseed_design.py",
    "verify_axial_multiseed_design.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    log_blocks = []
    for name in SCRIPTS:
        command = [sys.executable, str(REPO / "scripts" / name)]
        completed = subprocess.run(command, cwd=REPO, text=True, capture_output=True)
        block = [f"===== {name} =====", completed.stdout.rstrip()]
        if completed.stderr:
            block.extend(["STDERR:", completed.stderr.rstrip()])
        block.append(f"EXIT_CODE={completed.returncode}")
        log_blocks.append("\n".join(block))
        rows.append(
            {
                "script": name,
                "exit_code": completed.returncode,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
            }
        )
    (args.output_dir / "existing_verifier_runs.log").write_text(
        "\n\n".join(log_blocks) + "\n", encoding="utf-8"
    )
    payload = {
        "verifiers": rows,
        "overall_status": "PASS" if all(row["exit_code"] == 0 for row in rows) else "FAIL",
        "scope": "synthetic/static verifiers only; no training and no saved-model inference",
    }
    (args.output_dir / "existing_verifier_runs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    if payload["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
