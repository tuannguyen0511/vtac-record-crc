# VTaC record-level CRC for false VT alarm suppression

**Manuscript title:** Record-Level Sensitivity Guarantees for Suppressing False Ventricular Tachycardia Alarms: A Preregistered Evaluation With Annotator Labels and External Validation. **Status:** Under review at IEEE Journal of Biomedical and Health Informatics (IEEE JBHI).

This repository contains methods code and release metadata. It does **not**
redistribute raw waveforms, annotated signals, processed caches, or patient data.
The authors must review the release, complete the title/authors, and push it themselves.

## Data and licensing

- [VTaC v1.1](https://physionet.org/content/vtac/1.1/), CC BY-SA 4.0, DOI 10.13026/jg2e-tf67.
- [PhysioNet/CinC Challenge 2015 v1.0.0](https://physionet.org/content/challenge-2015/1.0.0/), ODC-By 1.0, DOI 10.13026/c9fg-a467.
- Download the two datasets from PhysioNet into `data_raw/vtac/1.1/` and
  `data_raw/challenge-2015/1.0.0/`; comply with their licenses.
- Only fitting/calibration record-ID lists are included in `splits/`.
  Use the official VTaC train/validation/test assignment; do not duplicate
  or reassign validation/test in this release.

## Environment

Python **3.11.9**; `python -m pip install -r requirements.txt`.
Reference hardware: **one NVIDIA Quadro M6000 (12 GB)**, **Intel Xeon E5-2680 v4**.
Runtime depends on CPU, GPU, storage and dataset preparation. No runtime claim
is provided for a fresh workstation.

1. Copy `configs/paths.example.yaml` to `configs/paths.yaml` and set
   relative locations for `data_raw`, `data_proc`, `outputs`, `paper_outputs`.
2. Obtain the two datasets separately. Never commit data, cache or scores.
3. Run the original approved stages in the following order:
   `T01b → T02b (cache) → T03 (25 checkpoints) → T05a → T05b → T06/T06b → T07 → paper/`.
4. Run `python scripts/verify_key_numbers.py` after arranging saved scores
   (see release assets). This verifier **does not** train or open T05b again.
5. Run `python paper/make_figures.py --out paper_outputs` to export five
   archived figures. This is an **export-only** smoke test, not a plot
   regeneration from raw measurements. Compare with final manuscript source.

**Important:** Original stage scripts contain fixed provenance and one-shot
integrity guards. Sanitized release configurations do not reproduce the original
file hashes. Do not disable checksum tests or rerun the one-time test to make
those scripts execute. Reconstruct frozen source artifacts and verify their
lineage before any end-to-end rerun. The verified saved-score workflow above
is the publication-ready inference and key-number check.

## Frozen CRC lock and one-time test

- `configs/final_lock.yaml` original SHA-256: `c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8`.
- `configs/final_lock.yaml` repository SHA-256: `a0afbb9a768d0b1a6304288f92d5d28c3fe14cca823bd6b054144a81336d0694`.
- Hashes may differ after sanitizing machine-dependent locations.
  All non-path scientific values must be unchanged; compare with the
  `P05-T08.md` build audit.
- The approved ensemble is `sigmoid(mean(seed_logits))`, **not**
  `mean(sigmoid(seed_logits))`.
- `src/t05b_final_test.py` opens the holdout test exactly once, with
  frozen `final_lock.yaml` and `reports/TEST_OPENED.flag`.
  **Do not run this script when verifying the published release.**

## Paper item mapping

The following entries are placeholders for the **authors to confirm** against
final `main.tex` and archived manuscript figures. Do not claim that every
item is regenerated until its source is verified.

| Paper item | Script | Output | Provenance |
|---|---|---|---|
| Table I | `paper/make_supplement_tables.py` | `paper_outputs/Table_I` | Archived-artifact export; verify against `main.tex` |
| Table II | `paper/make_supplement_tables.py` | `paper_outputs/Table_II` | Archived-artifact export; verify against `main.tex` |
| Table III | `paper/make_supplement_tables.py` | `paper_outputs/Table_III` | Archived-artifact export; verify against `main.tex` |
| Fig. 1 | `paper/make_figures.py` | `paper_outputs/Fig._1` | Archived-artifact export; verify against `main.tex` |
| Fig. 2 | `paper/make_figures.py` | `paper_outputs/Fig._2` | Archived-artifact export; verify against `main.tex` |
| Fig. 3 | `paper/make_figures.py` | `paper_outputs/Fig._3` | Archived-artifact export; verify against `main.tex` |
| Fig. 4 | `paper/make_figures.py` | `paper_outputs/Fig._4` | Archived-artifact export; verify against `main.tex` |
| Fig. 5 | `paper/make_figures.py` | `paper_outputs/Fig._5` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S1 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S1` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S2 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S2` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S3 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S3` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S4 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S4` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S5 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S5` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S6 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S6` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S7 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S7` | Archived-artifact export; verify against `main.tex` |
| Supplementary Table S8 | `paper/make_supplement_tables.py` | `paper_outputs/Supplementary_Table_S8` | Archived-artifact export; verify against `main.tex` |
| Supplementary Fig. S1 | `paper/make_figures.py` | `paper_outputs/Supplementary_Fig._S1` | Archived-artifact export; verify against `main.tex` |
| Supplementary Fig. S2 | `paper/make_figures.py` | `paper_outputs/Supplementary_Fig._S2` | Archived-artifact export; verify against `main.tex` |
| Supplementary Fig. S3 | `paper/make_figures.py` | `paper_outputs/Supplementary_Fig._S3` | Archived-artifact export; verify against `main.tex` |

## Pretrained checkpoints and saved scores

The 25 model checkpoints (A–E × seeds 0–4, approximately 4.5 MB each;
approximately 113 MB total) and saved logits are staged **outside this repo**
in `../release_assets/`. Upload them as a separate GitHub Release v1.0.0 asset,
not as Git commits. Each score table contains only `idx`, `logit` and,
where applicable, `seed`. The saved-score verifier uses the original cached
metadata and separately downloaded Challenge 2015 labels.

**Asset staging:** 25 checkpoint files,
52 score parquet files.
The published ZIP should carry `release_assets/SHA256_MANIFEST.json`.

## Privacy and limitations

No event waveform, E8 missed-case image, Windows user path, email or author
portrait is included. Avoid disclosing record identifiers in logs or issue
reports. Saved-score test/C2015 results are POST-HOC.

## Code availability

Code and archived model weights will be published by the authors after
manual inspection. GitHub URL: TODO. Zenodo DOI: TODO (optional).

### Checkpoint release integrity

The 25 public checkpoints were reserialized solely to redact Windows
workstation paths embedded in metadata. Their published SHA-256 hashes differ
from the original frozen checkpoint hashes. The release asset manifest records
both hashes. Tensor values and nonpath metadata were verified to match the
SHA-verified originals. No model was retrained. Original `final_lock.yaml`
values remain frozen.

### Optional external resource for the T01 downloader

The original T01 downloader contained one Windows path outside the P05
workspace. The public copy uses `configs/paths.yaml:t01_external_resource`,
which defaults to the relative placeholder `external_resource`. Set it to the
corresponding local file or directory if executing this legacy T01 branch.
Its role was not independently validated for end-to-end T01 portability.
