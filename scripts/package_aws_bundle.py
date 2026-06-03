from pathlib import Path
import shutil

project_root = Path(r"C:\Users\92127\OneDrive - UNSW\project_linpeng\code")
project_parent = project_root.parent
bundle_root = project_parent / "aws_upload_bundle"
code_bundle = bundle_root / "bhsd-nnunet"

exclude_top = {
    ".git",
    ".github",
    "nnUNet_data",
    "aws_upload_bundle",
    "results",
    "__pycache__",
    "SMOKE_TEST.md",
    "README.md",
    ".gitignore",
    "docs",
    "bhsd_spacing_summary.csv",
}
exclude_dir_names = {"__pycache__"}
exclude_suffixes = {".pyc", ".pyo", ".ps1", ".md", ".html"}
exclude_file_names = {"package_aws_bundle.py", "package_aws_bundle.ps1", "python"}
exclude_path_names = set()

if bundle_root.exists():
    shutil.rmtree(bundle_root)

code_bundle.mkdir(parents=True, exist_ok=True)

for item in project_root.iterdir():
    if item.name in exclude_top:
        continue
    dst = code_bundle / item.name
    if item.is_dir():
        shutil.copytree(
            item,
            dst,
            ignore=shutil.ignore_patterns(
                '__pycache__', '*.pyc', '*.pyo', '*.ps1', '*.md', '*.html',
                'package_aws_bundle.py', 'package_aws_bundle.ps1'
            )
        )
    else:
        if item.name in exclude_file_names or item.suffix in exclude_suffixes or item.name in exclude_path_names:
            continue
        shutil.copy2(item, dst)

for path in sorted(code_bundle.rglob('*'), reverse=True):
    if path.is_dir() and path.name in exclude_dir_names:
        shutil.rmtree(path)
    elif path.is_file() and (
        path.suffix in exclude_suffixes
        or path.name in exclude_file_names
        or path.name in exclude_path_names
    ):
        path.unlink()

upload_md = bundle_root / 'UPLOAD_TO_AWS.md'
upload_md.write_text(
    "AWS upload bundle created successfully.\n\n"
    "Upload only:\n"
    "- bhsd-nnunet  -> ~/projects/bhsd-nnunet\n\n"
    "This is the minimal server runtime bundle.\n"
    "Excluded: docs, Markdown files, Windows helpers, smoke-test notes, local packaging helpers.\n\n"
    "Then on the server run:\n\n"
    "cd ~/projects/bhsd-nnunet\n"
    "conda env create -f environment.yml || conda env update -f environment.yml\n"
    "conda activate bhsd-nnunet\n"
    "export PROJECT_ROOT=~/projects/bhsd-nnunet\n"
    "export nnUNet_raw=~/data/nnUNet_data/nnUNet_raw\n"
    "export nnUNet_preprocessed=~/data/nnUNet_data/nnUNet_preprocessed\n"
    "export nnUNet_results=~/data/nnUNet_data/nnUNet_results\n"
    "python scripts/install_extension.py\n",
    encoding='utf-8'
)

print(bundle_root)
print(sum(1 for _ in code_bundle.rglob('*') if _.is_file()))
