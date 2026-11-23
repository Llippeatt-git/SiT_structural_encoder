# SiT Structural Encoder — sMRI Surface Autoencoder

This repo is the **structural MRI (sMRI) branch** of a lab project on depression. The project's goal is to translate between modalities: each modality (tabular behavioural data, sMRI, …) gets its own encoder that produces a latent space, and the latent spaces are then fused so one modality can be translated into another.

This repo builds a **transformer autoencoder for reconstruction** using cortical surface data from HCP Young Adult (HCP-YA). The encoder is a Surface Vision Transformer (SiT), forked from [metrics-lab/surface-vision-transformers](https://github.com/metrics-lab/surface-vision-transformers), with a transformer decoder added. Left-hemisphere surfaces are split into patches on an icosphere, compressed to a single latent vector, and reconstructed from it.

> Status: work in progress. See `weekly_log.md` for the development history.

---

## Repository structure

Folders and files named `LEGACY` / `-LEGACY` are left out. They hold code from the original SiT/MS-SiT fork that isn't part of the active pipeline.

```
SiT_structural_encoder/
├── config/
│   └── hparams_recon.yml                 # Hyperparameters for reconstruction training (model, optimiser, paths)
├── models/
│   └── sit_recon_encoder.py              # SiT encoder + transformer decoder (autoencoder)
├── tools/
│   └── reconstruction_pipeline_v2.py     # Main training / validation / test script
├── utils/
│   ├── brain_ico_downsample_and_save_raw.ipynb  # Builds patched train/val/test .npy arrays from ico6 GIFTIs
│   ├── utils.py                          # Dataloaders, demeaning, per-subject correlation, GIFTI + scatter helpers
│   ├── recon_save_as_gifti.py            # Runs a trained model on the test set; saves high/median/low recons
│   ├── reinflate_recon.sh                # SLURM: runs recon_save_as_gifti.py, then resamples outputs to HCP 32k sphere
│   ├── demean_pipeline.py                # Saves one subject at each demeaning stage (for inspection)
│   ├── demeaning_snapshot.sh             # SLURM: runs demean_pipeline.py, then resamples to HCP 32k sphere
│   └── TESTING/                          # Scratch notebooks (recon-TESTING, demean-TESTING)
├── patch_extraction/
│   └── triangle_indices_ico_6_sub_ico_{1..5}.csv  # ico6 vertex indices per sub-ico patch
├── surfaces/
│   ├── ico-{0..6}.L.surf.gii, ico-6.R.surf.gii    # Template icospheres
│   ├── img_indices_40962.npy, img_weights_40962.npy
│   └── metric_resample.py, metric_resample_labels.py  # Sphere resampling helpers (not used by the active pipeline)
├── launchpad.sh                          # SLURM: launches reconstruction_pipeline_v2.py
├── requirements.txt
├── weekly_log.md                         # Development log
└── README.md                             # Original upstream SiT README
```

---

## Setup & dependencies

`requirements.txt` lists:

```
pyyaml
timm
pandas
einops
vit-pytorch
nibabel
tensorboard
```

```bash
pip install -r requirements.txt
```

The active code also imports these packages, which are **not** in `requirements.txt`:

| Package | Used in |
| --- | --- |
| `torch` | everywhere |
| `numpy` | everywhere |
| `scipy` | `utils/utils.py` (`stats.pearsonr(..., axis=1)`), `tools/reconstruction_pipeline_v2.py` |
| `matplotlib`, `seaborn` | `utils/utils.py` |
| `scikit-learn` | `brain_ico_downsample_and_save_raw.ipynb` (`train_test_split`) |

TODO: pin versions and add the missing packages to `requirements.txt`. TODO: state the Python version (the `__pycache__` files suggest CPython 3.12).

**External tools:**
- [Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench) (`wb_command`). The two `.sh` scripts load it with `module load workbench`.
- The `.sh` scripts are SLURM batch scripts written for the WashU CHPC cluster (`--partition=tier2_cpu`, `--account=janine_bijsterbosch`).

**Hard-coded paths:** most scripts point at absolute CHPC paths under `/ceph/chpc/shared/janine_bijsterbosch_group/lucas_scratch/...`. Running anywhere else means editing them (see [Known limitations](#known-limitations)).

---

## Data requirements

- **Source:** HCP-YA (S1200), left hemisphere only.
- **Input to preprocessing** (under `data_path` = `.../lucas_scratch/HCP_YA`):
  - `resampled_icos/<subject_id>/<subject_id>_ico6.L.shape.gii`: per-subject metric already resampled to ico6 (40,962 vertices). TODO: document which script produced these and which metric they hold (`reinflate_recon.sh` labels its outputs `corrThickness`).
  - `unrestricted_hcp_freesurfer.csv`: the first two columns are used as `[subject_id, sex]`.
  - `HCP_YA_subjects_2026_10_22_12_06_42.csv`: restricted HCP data; columns `[0, 5, 6, 7]` are read as `[subject_id, family_id, mother_id, father_id]`.
- **For reinflation:** native HCP spheres at `/ceph/chpc/rcif_datasets/hcpya/S1200/<id>/MNINonLinear/fsaverage_LR32k/<id>.L.sphere.32k_fs_LR.surf.gii`.

HCP restricted data has access rules. Don't commit any data files (`data/*` is already in `.gitignore`).

---

## Pipeline overview

```
ico6 .shape.gii per subject
        │  utils/brain_ico_downsample_and_save_raw.ipynb
        ▼
{train,validation,test}_L_{data,labels,subjects}.npy   (data: [B, 1, 320, 153])
        │  tools/reconstruction_pipeline_v2.py   (via launchpad.sh)
        ▼
<logs>/<date>-epoch_..-LR_..-dims_..-optim_../  best_model.pth, model_final.pth, hparams.yml, tensorboard events
        │  utils/reinflate_recon.sh  →  utils/recon_save_as_gifti.py
        ▼
high / median / low-correlation recon, target, delta  →  ico6 .shape.gii  →  32k_fs_LR .shape.gii (view in Workbench)
```

## Usage

### 1. Build patched train/val/test arrays

Open and run `utils/brain_ico_downsample_and_save_raw.ipynb` from top to bottom. Edit `data_path` and `patch_extraction_path` in cell 3 if needed. The notebook:

1. Lists the numeric subject folders in `resampled_icos/`.
2. Splits **by family**: 70% train, then the remainder split evenly into test and validation (`preprocess_and_sort(scans_ids, family_data, 0.7, 0.15)`), so siblings and twins never cross splits. `train_test_split` isn't seeded, so the split changes on every run.
3. Patches each subject's ico6 data onto the ico2 grid (320 patches × 153 vertices) using `patch_extraction/triangle_indices_ico_6_sub_ico_2.csv`.
4. Saves the following to `data_path`:
   - `{train,validation,test}_L_data.npy`: shape `[N, 1, 320, 153]`
   - `{train,validation,test}_L_labels.npy`: sex (1 = F), shape `[N, 1]` (loaded but not used for reconstruction)
   - `{train,validation,test}_L_subjects.npy`: HCP subject IDs, shape `[N, 1]`

The saved output shows 1113 subjects → train 772 / test 167 / validation 174.

### 2. Train the autoencoder

On the cluster:

```bash
sbatch launchpad.sh
```

This `cd`s into `tools/` and runs:

```bash
python3 ./reconstruction_pipeline_v2.py --config ../config/hparams_recon.yml
```

Run it from `tools/`: the script adds `./` and `../` to `sys.path` to import `models.` and `utils.`. `--config` defaults to `../config/hparams_recon.yml`.

### 3. Export and reinflate reconstructions for visualization

Edit `recon_dir` (output folder) and `model_path` (a training run folder containing `best_model.pth` and `hparams.yml`) at the top of `utils/reinflate_recon.sh`, then run it from `utils/`:

```bash
cd utils
sbatch reinflate_recon.sh
```

The script calls:

```bash
python3 ./recon_save_as_gifti.py --out_dir "<recon_dir>" --model_dir "<model_path>"
```

It then uses `wb_command -metric-resample ... BARYCENTRIC` to resample the ico6 GIFTIs onto each subject's native HCP `32k_fs_LR` sphere, and copies that subject's sphere and midthickness surfaces into the output folder so they can be viewed in Workbench.

### Optional: inspect demeaning on one subject

```bash
cd utils
sbatch demeaning_snapshot.sh     # edit `index` and `output_dir` inside first
# runs: python3 ./demean_pipeline.py --out_dir <output_dir> --subject_index_get <index>
```

This saves the chosen test subject at three stages (raw, per-subject normalized, then fully demeaned against the train vertex-wise mean) and resamples each one to the 32k sphere.

---

## Priority files

### `utils/brain_ico_downsample_and_save_raw.ipynb`
- **Purpose:** preprocessing. Downsamples/patches ico6 surface metrics onto the ico2 patch grid and writes family-aware train/val/test splits.
- **Inputs:** `resampled_icos/<id>/<id>_ico6.L.shape.gii`, the two HCP CSVs listed above, and `triangle_indices_ico_6_sub_ico_2.csv`.
- **Outputs:** the nine `*_L_{data,labels,subjects}.npy` files in `data_path`.
- **Key function:** `preprocess_and_sort(ids, family_ids, train_split, test_split)`.

### `models/sit_recon_encoder.py`
- **Purpose:** defines `SiT`, a ViT-style surface autoencoder.
- **Encoder (`encode`):** `[B, C, N, V]` → patch linear embedding → prepend CLS token + learned positional embedding → `vit_pytorch.vit.Transformer` (`depth`) → CLS token (or mean, with `pool='mean'`) → `LayerNorm` + `Linear` → latent `[B, latent_dim]`.
- **Decoder (`decode`):** latent → `Linear` to `dim` → concatenated after `num_patches` learned query tokens + decoder positional embedding → `Transformer` (`dec_depth`) → drop the latent token → `Linear` to `patch_dim` → reshape to `[B, 1, N, V]`.
- **`forward(x)`:** `decode(encode(x))`.
- With the default config (`dim 192`, `depth 12`, `dec_depth 4`, `heads 6`, `latent_dim 768`, ico2 grid), the input is `[B, 1, 320, 153]` and the latent is 768-d.

### `tools/reconstruction_pipeline_v2.py`
- **Purpose:** trains, validates and tests the autoencoder on reconstruction (target = input).
- **Inputs:** `--config` YAML. Data comes from `get_dataloaders()` in `utils/utils.py`.
- **Loss:** `mse` or `l1` (`training.loss`). **Optimisers:** `Adam`, `AdamW`, `SGD`.
- **Metrics:** MAE and mean per-subject Pearson correlation (`calculate_correlation`), logged to TensorBoard as `loss/`, `mae/`, `corr/` × `train`/`val`.
- **Validation:** every `val_epoch` epochs and on the last epoch. The best model is chosen by **lowest val MAE**.
- **Outputs**, in `<folder_to_save_model>/<YYYY-MM-DD-HH:MM:SS>-epoch_<E>-LR_<LR>-dims_<dim>-optim_<opt>/`:
  - `best_model.pth`: best-val-MAE weights (plain `state_dict`)
  - `model_final.pth`: last-epoch weights
  - `hparams.yml`: the config plus a `results` block (`best_mae`, `best_corr`, `best_epoch`, `test_accuracy` [= test MAE], `test_correlation`)
  - TensorBoard event files
- Reloads `best_model.pth` at the end and prints test MAE and correlation.

### `utils/recon_save_as_gifti.py`
- **Purpose:** runs a trained model on the test set, ranks subjects by reconstruction correlation, and exports the highest, median and lowest subjects.
- **CLI:** `--out_dir` (required), `--model_dir` (required; must contain `best_model.pth` and `hparams.yml`).
- **Outputs** (in `--out_dir`):
  - `bulk_data/targets_data.npy`, `predicted_data.npy`, `subjects_data.npy`, `subject_correlations.csv`
  - `recons_snapshot/{high,median,low}_{target,recon,delta}_subj-<id>-.npy`, plus a scatter plot `{name}_correlation_subj-<id>-` (written by `display_single_scatter`)
  - `original_*`, `reconstructed_*`, `recon_delta_*` `.L.shape.gii` at ico6 (40,962 vertices), written at the top level of `--out_dir` by `save_reconstruction_mae_test`

### `utils/reinflate_recon.sh`
- **Purpose:** SLURM wrapper around `recon_save_as_gifti.py`, followed by Workbench resampling to native HCP space.
- **Inputs:** `recon_dir` and `model_path` (set inside the script), `surfaces/ico-6.L.surf.gii`, and the HCP S1200 per-subject 32k spheres.
- **Outputs:** in `recon_dir`, `<prefix>_<id>_corrThickness_32k_fs.L.shape.gii` plus that subject's `L.sphere` and `L.midthickness` 32k surfaces. The original ico6 GIFTIs are moved to `recon_dir/original_icos/`.

---

## Supporting files (skimmed)

- **`utils/utils.py`:** `demean()` (per-subject z-score, then subtract the train vertex-wise mean; **currently commented out** in `get_dataloaders`), `get_dataloaders()`, `calculate_correlation()`, `save_gifti()`, `save_reconstruction_mae_test()`, `display_single_scatter()`.
- **`utils/demean_pipeline.py` / `utils/demeaning_snapshot.sh`:** see "Optional" above.
- **`surfaces/metric_resample*.py`:** sphere resampling helpers from the upstream repo. The priority files don't import them.

---

## Results so far

From `weekly_log.md` only. These haven't been re-verified in this README.

- Without demeaning: about 0.92 per-subject correlation between reconstruction and target. Further inspection (`utils/TESTING/recon-TESTING.ipynb`) suggested the model **had learned the group mean** (near-identical reconstructions across subjects).
- With per-vertex demeaning and per-subject normalization: training **did not converge** (correlation about 0.0).

TODO: record the current best run (run folder, config, val/test MAE and correlation).

---

## Known limitations

- **Left hemisphere only**, 1 channel. Right-hemisphere code is commented out.
- **Hard-coded cluster paths.** `reconstruction_pipeline_v2.py` ignores `config['data']['path_to_workdir']` and uses a hard-coded `data_path`. `recon_save_as_gifti.py`, `utils.save_reconstruction_mae_test`, `demean_pipeline.py` and the notebook do the same.
- **The non-seeded split** in the notebook means the train/val/test arrays can't be reproduced exactly.
- **Demeaning is off.** `config.training.demeaned: True` isn't read anywhere, and the `demean(...)` call in `get_dataloaders` is commented out.
- `get_dataloaders` uses `num_workers=16`, but `launchpad.sh` requests only 8 CPUs, on a CPU partition (`tier2_cpu`) with no GPU.
- The model's decoder always reshapes its output to 1 channel, whatever `num_channels` is.

---

## Credits

Built on the Surface Vision Transformer (Dahan et al., MIDL 2022) from [metrics-lab/surface-vision-transformers](https://github.com/metrics-lab/surface-vision-transformers), which uses [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch). Citation details are in the original `README.md`.
