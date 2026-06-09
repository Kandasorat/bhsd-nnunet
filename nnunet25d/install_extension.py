from pathlib import Path
import shutil
import site

import nnunetv2


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")


def main() -> None:
    package_root = Path(__file__).resolve().parent
    nnunet_root = Path(nnunetv2.__path__[0])
    site_packages_root = Path(site.getsitepackages()[0])

    package_dst = site_packages_root / "nnunet25d"
    package_dst.mkdir(parents=True, exist_ok=True)
    for src in package_root.rglob("*.py"):
        if "__pycache__" in src.parts:
            continue
        copy_file(src, package_dst / src.relative_to(package_root))

    copy_file(package_root / "dataloader_25d.py", nnunet_root / "training" / "dataloading" / "dataloader_25d.py")
    copy_file(package_root / "trainer_25d.py", nnunet_root / "training" / "nnUNetTrainer" / "trainer_25d.py")
    copy_file(package_root / "trainer_25d_5slice.py", nnunet_root / "training" / "nnUNetTrainer" / "trainer_25d_5slice.py")
    copy_file(package_root / "trainer_spacing_aware.py", nnunet_root / "training" / "nnUNetTrainer" / "trainer_spacing_aware.py")
    copy_file(
        package_root / "trainer_25d_feature_fusion_shim.py",
        nnunet_root / "training" / "nnUNetTrainer" / "trainer_25d_feature_fusion.py",
    )

    print("")
    print("2.5D extension install finished.")
    print(f"Installed helper package to: {package_dst}")


if __name__ == "__main__":
    main()
