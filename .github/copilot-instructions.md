Project: BHSD dataset sanity/check script

Purpose
- Short: help an AI contributor quickly understand and work on the dataset sanity-check script in this repo.

Big picture
- Single-script repo: the main work is in [bhsd_code_check.py](bhsd_code_check.py#L1-L200). It scans a local BHSD dataset directory, pairs image (`images/*.nii*`) and label (`ground truths/*.nii*`) volumes, and writes `bhsd_sanity_report.csv` in the BHSD root.
- Data flow: disk (.nii/.nii.gz) -> nibabel load -> numpy arrays -> pandas DataFrame -> CSV output.
- Why structure looks like this: script is a quick local sanity utility (hard-coded `BHSD_ROOT`) rather than a packaged library or CLI.

How to run (developer workflow)
- Recommended: create/activate a Python venv and install dependencies: `pip install numpy pandas nibabel`.
- Run locally on Windows from repository root:
  - `python bhsd_code_check.py`
- If dataset is elsewhere, update `BHSD_ROOT` in [bhsd_code_check.py](bhsd_code_check.py#L1-L40) or set up a small wrapper that injects the path (no CLI/arg parsing present).

Key project-specific patterns (copyable examples)
- File pairing: filenames are matched by a `stem()` helper that strips `.nii` and `.nii.gz`. Keep this exact logic when adding new matching code: see [bhsd_code_check.py#L6-L18](bhsd_code_check.py#L6-L18).
- Image/label checks: equality of `shape` is used to mark `SHAPE_MISMATCH` and `fg_ratio = (lbl>0).sum() / lbl.size` is used to measure foreground proportion — preserve numeric types when porting (cast to float for JSON/CSV compatibility).
- Use `nib.load(...).get_fdata()` for safe numeric arrays (script expects float arrays); follow this pattern when adding processing stages.

Dependencies & environment
- Python 3.8+ recommended.
- Python packages: `numpy`, `pandas`, `nibabel`, `json` (stdlib), `glob`/`os` (stdlib).

Conventions & gotchas
- Paths are Windows-style by default (`BHSD_ROOT` is a Windows path in the script). Avoid assuming POSIX paths unless you add cross-platform handling.
- Dataset subfolders: `images` and `ground truths` (note the space in `ground truths`) — reference exact names when writing code that lists or creates dirs.
- Output CSV: `bhsd_sanity_report.csv` is written into `BHSD_ROOT` unconditionally; avoid overwriting without explicit intent.

Integration points worth checking first
- Any change that touches I/O should reference the `BHSD_ROOT` value and preserve the `stem()` matching behavior.
- If adding unit tests or CI, mock file system or provide a tiny sample `.nii` (nibabel-readable) fixture; currently there is no test harness or CI configured.

When editing or extending
- Prefer small, local edits: add CLI args (argparse) only if you update run docs and tests.
- Keep numeric summaries (label values, fg_ratio) as small JSON-serializable primitives before writing to CSV/JSON.

Next steps / questions for the maintainer
- Do you want `BHSD_ROOT` parameterized (env var / CLI)?
- Should we add a small test dataset and a CI job that runs this script on Windows runners?

If anything here is incomplete or unclear, tell me which workflows (tests, CI, packaging) you care about and I will expand this file.
