# Depression Decoding

High-level development log.

## Log

- [11 Oct – 17 Oct 2026: Getting started](#11-oct--17-oct-2026-getting-started)
- [18 Oct – 24 Oct 2026: Moving to reconstruction](#18-oct--24-oct-2026-moving-to-reconstruction)
- [25 Oct – 31 Oct 2026: More reconstruction](#25-oct--31-oct-2026-more-reconstruction)
- [1 Nov – 7 Nov 2026: Starting visualization](#1-nov--7-nov-2026-starting-visualization)
- [8 Nov – 14 Nov 2026: Demeaning & taking point](#8-nov--14-nov-2026-demeaning--taking-point)

---

## 11 Oct – 17 Oct 2026: Getting started

- Forked the surface vision transformer from the Dahan Metrics Lab: <https://github.com/metrics-lab/surface-vision-transformers>
- Removed MS-SiT code from the training pipeline, focusing solely on SiT.
- Replaced `get_dataloaders` with custom dataloader code tailored to the HCP-YA data structure.
- Using most of the existing training infrastructure (transformer encoder unchanged), trained a model for sex prediction on HCP-YA.
  - Loss function: binary cross entropy
  - Performance metrics: AUC and accuracy
- Model trained for 100 epochs. Results: **AUC 0.807, accuracy 0.701**.
  - Conferred with Janine; confirmed performance is consistent with what we would expect.
- Saved training pipeline as `sex_encoder.py`.

## 18 Oct – 24 Oct 2026: Moving to reconstruction

- Developed a script to convert HCP-YA data into icospheres and sort them into train/test/val bins.
  - `Ico_downsample.ipynb` saves the train/test/val splits.
- **Currently:** splitting by subject ID, so twins with different IDs may end up in different bins.
  - **TO BE RESOLVED**
- Removed the `mlp_head` output on the encoder and added a decoder found online.
  - Saved as `sit_recon_encoder.py` in the `tools` folder.
- Reverted the training pipeline for reconstruction.
  - Loss function: MSE
  - Performance metric: MAE
- Saved a copy of the training pipeline as `reconstruction_pipeline.py`.
- Tested/debugged the naive reconstruction pipeline and obtained **MAE ~0.3** using AdamW and default settings (pulled from Dahan).

## 25 Oct – 31 Oct 2026: More reconstruction

- Implemented family-based splitting in `ico_downsample.ipynb`, so twins/siblings are always put in the same bin.
- Implemented subject-wise correlation in `reconstruction_pipeline` as a more interpretable metric for success.
- Retest shows **~0.92 correlation** between reconstructions and targets (same hyperparameters as the previous week).
- Began work on a visualization pipeline, taking reconstruction data and formatting/resampling it into an icosphere that can be displayed in Workbench.

## 1 Nov – 7 Nov 2026: Starting visualization

- Finished the visualization pipeline.
  - `Recon_save_as_gifti.py` takes a trained model on the test data and returns the ico6 CIFTI file for the reconstructions with the lowest, highest, and median correlations.
  - `Reinflate_recon.sh` adds structure to the outputs of `recon_save_as_gifti`, so they can be displayed on an ico6 surface in Workbench.
- Ran the visualization pipeline and intuited that the model had learned the mean.
- Further testing (`recon_testing.ipynb`) confirms that the model has learned the mean: all reconstructed values are the same.
- Identified demeaning as a possible resolution; implemented it at the per-vertex level alongside per-subject normalization (for scaling effects).
- Tested the model with the same parameters on the demeaned sample.
  - Failed to converge, correlation ~0.0.

## 8 Nov – 14 Nov 2026: Demeaning & taking point

- Restructured code and moved peripheral functions into `utils.py` (including `get_dataloaders`, correlation calculation, etc.).
- Revised the visualization pipeline, adding a processing step that resamples from ico6 back onto the brain structure found in the HCP-YA dataset.
- Developed code to visualize the correlation for a single target/prediction pair on a scatterplot. Also stored in `utils.py`.
- Developed a pipeline to view single-subject demeaning.
  - `Demeaning_pipeline.py` selectively applies different steps of the demeaning process to a chosen subject's data, then saves it out as a CIFTI.
  - `Demeaning_snapshot.sh` adds structure to the file output, resamples it to brain structure, and stores it in a folder.

---

## Files referenced

| File | Purpose |
| --- | --- |
| `sex_encoder.py` | SiT training pipeline for sex prediction |
| `Ico_downsample.ipynb` | Converts HCP-YA data to icospheres and creates train/test/val splits |
| `sit_recon_encoder.py` | Encoder with decoder attached (in `tools`) |
| `reconstruction_pipeline.py` | Reconstruction training pipeline (MSE / MAE / correlation) |
| `Recon_save_as_gifti.py` | Saves lowest, highest, and median correlation reconstructions as ico6 files |
| `Reinflate_recon.sh` | Adds structure so outputs display on an ico6 surface in Workbench |
| `recon_testing.ipynb` | Tests confirming the model learned the mean |
| `utils.py` | Shared helpers (dataloaders, correlation, scatterplot) |
| `Demeaning_pipeline.py` | Applies demeaning steps to a single subject and saves a CIFTI |
| `Demeaning_snapshot.sh` | Adds structure, resamples to brain structure, and stores the result |
