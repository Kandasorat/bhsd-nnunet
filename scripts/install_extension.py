from pathlib import Path
import subprocess
import sys


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(project_root / "nnunet25d" / "install_extension.py")], check=True)


if __name__ == "__main__":
    main()
