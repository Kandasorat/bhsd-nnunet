from pathlib import Path
import shutil

import nnunetv2


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    nnunet_root = Path(nnunetv2.__path__[0])

    copy_file(
        project_root / "dataloader_25d.py",
        nnunet_root / "training" / "dataloading" / "dataloader_25d.py",
    )
    copy_file(
        project_root / "trainer_25d.py",
        nnunet_root / "training" / "nnUNetTrainer" / "trainer_25d.py",
    )

    print("")
    print("Install finished.")
    print("Use the trainer with:")
    print("  nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D")
    print("or")
    print("  nnUNetv2_train Dataset001_BHSD 2d 0 -tr nnUNetTrainer_25D_5Slice")


if __name__ == "__main__":
    main()
