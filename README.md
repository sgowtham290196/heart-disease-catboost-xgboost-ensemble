# Heart Disease CatBoost + XGBoost Ensemble

This project trains a CatBoost + XGBoost ensemble for the Kaggle dataset and generates a submission file.

## What It Does
- Loads `train.csv`, `test.csv`, and `sample_submission.csv`
- Encodes target (`Absence`/`Presence`) to 0/1
- Trains CatBoost and XGBoost with stratified CV
- Optimizes blend weights on OOF AUC
- Writes Kaggle-ready submission (`id,Heart Disease`)

## Project Files
- `train_cat_xgb_ensemble.py`: end-to-end training and submission script
- `requirements.txt`: Python dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python train_cat_xgb_ensemble.py \
  --train-path ../train.csv \
  --test-path ../test.csv \
  --sample-submission-path ../sample_submission.csv \
  --submission-out submission.csv \
  --report-out report.json
```

## Output
- `submission.csv`: file ready for Kaggle upload
- `report.json`: model AUCs, fold metrics, blend weights

## Notes
- This script uses only CatBoost and XGBoost.
- `id` is excluded from training features.
