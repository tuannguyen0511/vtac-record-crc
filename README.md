# VTaC record-level CRC for false VT alarm suppression

**Manuscript title:** Record-Level Sensitivity Guarantees for Suppressing False Ventricular Tachycardia Alarms: A Pre-Specified, One-Shot Evaluation With Annotator Labels and External Validation. **Status:** Under review at IEEE Journal of Biomedical and Health Informatics (IEEE JBHI).

**Software version:** `1.1.0` (see `CITATION.cff`).

This repository contains methods code and release metadata. It does **not**
redistribute raw waveforms, annotated signals, processed caches, or patient data.

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
5. Run `python paper/make_figures.py --out paper_outputs` to generate five
   figures from the audited aggregate outputs. To export archived PDFs,
   run `python paper/export_archived_figures.py --out paper_outputs`.

**Important:** Original stage scripts contain fixed provenance and one-shot
integrity guards. Sanitized release configurations do not reproduce the original
file hashes. Do not disable checksum tests or rerun the one-time test to make
those scripts execute. Reconstruct frozen source artifacts and verify their
lineage before any end-to-end rerun. The verified saved-score workflow above
is the publication-ready inference and key-number check.

## Frozen CRC lock and one-time test

- `configs/final_lock.yaml` original SHA-256: `c09b6cd952319950f65b57e64086ab691946ccabac84adb106efb2bc765299c8`.
- `configs/final_lock.yaml` repository SHA-256: `d99cf05b94b82cf7e76b90e3b9fa093623611578c9bb6a83970452d34ba905b6`.
- Hashes may differ after sanitizing machine-dependent locations.
  All non-path scientific values must be unchanged; compare with the
  `P05-T08.md` build audit.
- The approved ensemble is `sigmoid(mean(seed_logits))`, **not**
  `mean(sigmoid(seed_logits))`.
- `src/t05b_final_test.py` opens the holdout test exactly once, with
  frozen `final_lock.yaml` and `reports/TEST_OPENED.flag`.
  **Do not run this script when verifying the published release.**

## Paper item mapping

The paths below were verified against the local frozen outputs and the manuscript.
"Generated" means the named script writes the artifact. "Values transcribed"
means the cited output supplies values entered in `main.tex`.

| Paper item | Generating script | Source in outputs/ | How produced |
|---|---|---|---|
| Table I (tab:variants) | `src/t05b_final_test.py` | `outputs/t05b/summary.json` | values transcribed from outputs/t05b/summary.json |
| Table II (tab:external) | `src/t05b_final_test.py` | `outputs/t05b/summary.json` | values transcribed from outputs/t05b/summary.json |
| Fig. 1 (pipeline schematic) | `paper/make_figures.py` | none (schematic; no output data input) | generated |
| Fig. 2 (CRC versus LTT) | `paper/make_figures.py` | `outputs/t05b/summary.json` | generated (numeric values transcribed from outputs/t05b/summary.json) |
| Fig. 3 (calibration stability) | `paper/make_figures.py` | `outputs/t06_hf1/A1_bootstrap.parquet` | generated |
| Fig. 4 (prevalence shift) | `paper/make_figures.py` | `outputs/t06/A4_prevalence.parquet` | generated |
| Fig. 5 (acquisition profiles) | `paper/make_figures.py` | outputs/t06b/trained_cross_profile.parquet; outputs/t06b/profile_crc.parquet | generated |
| Supplementary ablation table | `paper/make_supplement_tables.py` | outputs/t03/A_s0..A_s4/scores_val.parquet; outputs/t05a/ablation_summary.parquet | generated |
| Supplementary tables: event CRC, bSQI, latency, missed cases | `paper/make_supplement_tables.py` | outputs/t07/E3_crc_event_vs_record.parquet; outputs/t07/E3_thresholds.parquet; outputs/t07/E5_tertiles.parquet; outputs/t07/E7_latency.parquet; outputs/t07/E7_preprocessing_latency.parquet; outputs/t07/E8_missed_summary.parquet | generated |
| Fig. S1 (missed-case waveforms) | `src/t07_presubmission.py` | not distributed (waveforms of individual recordings; regenerate locally with src/t07_presubmission.py after downloading the data) | not distributed |
| Fig. S2 (missed-case waveforms) | `src/t07_presubmission.py` | not distributed (waveforms of individual recordings; regenerate locally with src/t07_presubmission.py after downloading the data) | not distributed |
| Fig. S3 (missed-case waveforms) | `src/t07_presubmission.py` | not distributed (waveforms of individual recordings; regenerate locally with src/t07_presubmission.py after downloading the data) | not distributed |
| Narrative: test/external performance, CRC and baseline comparisons | `src/t05b_final_test.py` | `outputs/t05b/summary.json` | values transcribed from outputs/t05b/summary.json |
| Narrative: supporting post-hoc sensitivity analysis | `src/t06_branch_a.py` | `outputs/t06/A4_prevalence.parquet` | values transcribed from outputs/t06/A4_prevalence.parquet |
| Narrative: supporting post-hoc sensitivity analysis | `src/t06_hf1.py` | `outputs/t06_hf1/A1_bootstrap.parquet` | values transcribed from outputs/t06_hf1/A1_bootstrap.parquet |

## Pretrained checkpoints and saved scores

The 25 model checkpoints (A–E × seeds 0–4, approximately 4.5 MB each;
approximately 113 MB total) and saved logits are staged **outside this repo**
in `../release_assets/`. These original weights are distributed as separate v1.0.0 release assets,
not as files tracked by Git. Each score table contains only `idx`, `logit` and,
where applicable, `seed`. Run `python scripts/verify_key_numbers.py --assets ../release_assets` to verify without the T02b cache. Challenge 2015 labels still come from downloaded data.

**Asset staging:** 25 checkpoint files,
52 score parquet files, and `labels_index.parquet` (calib/val/test labels and public VTaC alarm IDs).
The release asset inventory is recorded in `release_assets/SHA256_MANIFEST.json`.

## Privacy and limitations

No event waveform, E8 missed-case image, Windows user path, email or author
portrait is included. Avoid disclosing record identifiers in logs or issue
reports. Saved-score test/C2015 results are POST-HOC.

## Code availability
Code: https://github.com/tuannguyen0511/vtac-record-crc.
GitHub Release v1.0.0 retains the original `checkpoints.zip` and
`saved_scores.zip` assets. The v1.1.0 release package additionally
includes `checkpoints_t10.zip` for the post-hoc T10 analyses.
Zenodo DOI: not assigned.

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

### HF5 reproducibility commands

Run `python paper/make_supplement_tables.py --out paper_outputs/` to generate
`supp_tab_ablation.tex` and `supplement_tables.tex` from paths configured in
`configs/paths.yaml`. `paper/make_figures.py --out paper_outputs/` creates the
five scientific PDFs (and PNG previews). `paper/export_archived_figures.py`
only exports the frozen manuscript PDFs, without recomputing figure content.

The `--assets` verifier reads the 25-seed saved scores and the
`labels_index.parquet` file (public VTaC alarm identifiers, no waveforms).
Download Challenge 2015 separately and set `data_raw` in `paths.yaml`.
For local verification, no GPU is necessary. On a CPU-only computer, use
PyTorch 2.0.1 CPU wheels (e.g., install `torch==2.0.1` from the CPU index)
instead of `torch==2.0.1+cu117`; the locked saved-score verifier does not
run a neural network and does not require a GPU.

The checked `paths.yaml` may reference data directories outside the cloned
repository using relative paths. New users should copy
`configs/paths.example.yaml` and adjust all paths to their local installation.

### Figure 2 rounding correction

At CRC alpha=0.01 on Challenge 2015, the earlier figure source
displayed 13.4% false-alarm suppression. The verified count is
34/252 = 13.492063%, which rounds to **13.5%** at one decimal place.
The repository figure source uses 13.5%; the archived manuscript
PDFs and original frozen results are unchanged.

## Post-hoc analyses (T10/T11)

**Scope:** Every T10/T11 analysis is **POST-HOC**. Both the official test
labels and the Challenge 2015 labels had been opened before these analyses.
No post-hoc threshold or model comparison replaces the original one-shot test.
The T11 CALIB N5 correction uses the full floating-point CRC boundary and
does not edit the earlier archived T10 table.

| Manuscript item | Public aggregate artifact (all `results/t10/`) |
|---|---|
| Table II, mean recording-level FNR | `N8_record_fnr.csv` |
| Supplementary S10.1, second FCN backbone | `N3_fcn_crc_results.csv`, `N3_summary.json` |
| Supplementary S10.2, Fig. S4, suppression versus alpha | `N7_suppression_alpha.csv`, `N7_suppression_alpha.pdf` |
| Supplementary S10.3, soft-label mechanism | `N1_summary.json`, `N1_cdf.pdf` |
| Supplementary S10.4, matched-size control | `N11_evaluation.json`, `N11_plan.json` |
| Challenge score at frozen thresholds | `N4_challenge_score.json` |
| Fingerprint exclusion sensitivity | `N6_fingerprint.json` |
| Fractional CRC versus binary-order statistic | `N9_fractional_vs_binary.json` |
| Corrected N5 calibration record FNR | `N5_calib_corrected.json` |
| Wall-time provenance | `T10_stage_runtime.json` |

To check the original twelve locked quantities and the additional aggregate
T10/T11 tables, run `python scripts/verify_key_numbers.py` or, after placing
the published saved scores in a local directory,
`python scripts/verify_key_numbers.py --assets <local-saved-scores-dir>`.
The added checks are consistency checks of the published summary tables;
they do **not** independently recompute neural-network predictions. The
original twelve checks continue to use the saved logits and original labels.
Challenge 2015 labels must be obtained separately from the lawful public source.

N1 uses binary entropy of the cached mean vote probability, not the
unavailable categorical entropy of the original annotator responses.
Subgroup counts below 10 are redacted as `<10` (the whole-cohort FNR k/n
is a stated exception). No individual waveform, record or event identifier,
or pseudonym list is distributed with the new T10 summaries. The N11
plan's digest of pseudonym selections is excluded from the public copy.
The ten T10 checkpoint weights (5 FCN and 5 matched-size control) are staged
**outside the Git repository** as `release_assets/checkpoints_t10.zip`.
The additional T10 weights are packaged as the separate
`checkpoints_t10.zip` asset for v1.1.0. Original v1.0.0 assets remain separate.
