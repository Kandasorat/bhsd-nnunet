from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    parser = argparse.ArgumentParser(description="Refuse Stage3 training unless the frozen V100 gate still matches")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    if gate.get("gate_pass") is not True or gate.get("status") != "PASS":
        raise RuntimeError("V100 gate did not pass")
    if "V100" not in str(gate.get("device_name", "")).upper():
        raise RuntimeError("Gate was not measured on V100 hardware")
    checks = gate.get("checks", {})
    if not all(checks.get(name) is True for name in ("v100", "git_clean", "capacity_match", "latency_pass", "memory_pass")):
        raise RuntimeError(f"Incomplete V100 gate checks: {checks}")
    current_head = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_head != gate.get("git_head"):
        raise RuntimeError(f"Git HEAD changed after V100 gate: {current_head} != {gate.get('git_head')}")
    dirty = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
    ).splitlines()
    if dirty:
        raise RuntimeError(f"Working tree changed after V100 gate: {dirty}")
    config_hash = sha256(args.config)
    expected_hash = gate.get("config_hashes", {}).get(args.config.stem)
    if config_hash != expected_hash:
        raise RuntimeError(f"Config changed after V100 gate: {config_hash} != {expected_hash}")
    print(
        json.dumps(
            {
                "status": "PASS",
                "git_head": current_head,
                "gate": str(args.gate),
                "config": str(args.config),
                "config_sha256": config_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
