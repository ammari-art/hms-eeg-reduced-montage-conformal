# Learning from Expert Disagreement: Reduced-Montage EEG Classification of Harmful Brain Activity with Patient-Level Conformal Uncertainty

Code for the paper submitted to *Computers in Biology and Medicine* (2026).

The pipeline classifies six EEG patterns (seizure, LPD, GPD, LRDA, GRDA, other) on the public
[HMS – Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)
dataset. It compares training on expert vote distributions with majority labels, evaluates fixed and learned
reduced electrode montages, and computes naive, Mondrian, patient-weighted and patient-level conformal prediction sets.

## Main results (5-fold patient-grouped cross-validation, 17,089 recordings, 1,950 patients)

| Model | Channels | KL ↓ | Macro-AUROC ↑ |
|---|---|---|---|
| CNN–Transformer, vote-distribution (KL) loss | 18 | 0.586 ± 0.047 | 0.920 ± 0.014 |
| CNN–Transformer, majority-label (CE) loss | 18 | 0.607 ± 0.035 | 0.916 ± 0.015 |
| Temporal-chain (hairline) montage | 8 | 0.660 ± 0.047 | 0.902 ± 0.021 |
| Learned subset, k = 10 (6.8 distinct channels) | 6.8 | 0.642 ± 0.075 | 0.903 ± 0.022 |
| EEGNet | 18 | 0.917 ± 0.034 | 0.821 ± 0.033 |

Patient-level conformal prediction (α = 0.10) raised worst-decile patient coverage from 55.0% to 62.3%.

Public Kaggle notebook with the full run log: https://www.kaggle.com/code/ammaridr/notebook221ab9193b

## How to reproduce (Kaggle, free GPU)

1. Join the competition on Kaggle and accept its rules.
2. Create a new notebook from the competition page (this attaches the data), then **File → Import Notebook** and upload `hms_disagreement_reduced_montage.ipynb`.
3. **Settings → Accelerator → GPU T4 x2.**
4. Run a smoke test with `QUICK = True` (about 3 minutes).
5. Set `QUICK = False` and use **Save Version → Save & Run All (Commit)**. The full run (12 experiments × 5 folds) takes about 6 hours on one T4.
6. Tables are written to `/kaggle/working/results/` (`table1_dataset.json`, `table2_point_metrics.csv`, `table3_per_class_sensitivity.csv`, `table4_conformal.csv`, `table5_selected_channels.csv`, `table6_tests.csv`, `fig4_occlusion_full18_kl.csv`).

The data folder is detected automatically. The script also runs outside Kaggle if `CFG["data_dir"]` points to the
downloaded competition data (`pip install -r requirements.txt`).

## Files

| File | Content |
|---|---|
| `hms_disagreement_reduced_montage.ipynb` | Kaggle notebook (same code as the script) |
| `hms_disagreement_reduced_montage.py` | Plain Python script with all steps |
| `make_paper_figures.py` | Draws Figures 2 and 4 of the paper from the Kaggle results |

## Data

The EEG data are not redistributed here. Download them from Kaggle under the competition rules.

## Citation

```
[Authors]. Learning from Expert Disagreement: Reduced-Montage EEG Classification of Harmful Brain Activity
with Patient-Level Conformal Uncertainty. Computers in Biology and Medicine (submitted), 2026.
```

## License

MIT (code only). The dataset is subject to the Kaggle competition rules.
