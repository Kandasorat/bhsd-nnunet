from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "SHA256SUMS.txt"
    lines = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != output and "__pycache__" not in p.parts):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {len(lines)} checksums to {output}")


if __name__ == "__main__":
    main()
