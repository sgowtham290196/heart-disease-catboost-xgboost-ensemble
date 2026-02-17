# Heart Disease CatBoost + XGBoost Ensemble

End-to-end Kaggle tabular classification pipeline for predicting heart disease using a CatBoost + XGBoost ensemble with stratified cross-validation, out-of-fold (OOF) predictions, and AUC-based blend optimization.

## Dataset

This project uses the Kaggle Playground competition dataset:
- Competition: [Playground Series - Season 6, Episode 2](https://www.kaggle.com/competitions/playground-series-s6e2)
- Data files used:
  - `train.csv`
  - `test.csv`
  - `sample_submission.csv`

Target column:
- `Heart Disease` (`Absence` / `Presence`)

ID column:
- `id`

## Problem Type

- Binary classification
- Evaluation metric: ROC-AUC (same objective used for validation and blending)

## Model Approach

Base learners:
1. **CatBoostClassifier**
2. **XGBClassifier**

Both models are trained with **Stratified K-Fold CV** to preserve class proportions in each fold.

### Why this ensemble

- CatBoost is strong on mixed tabular features and robust with minimal preprocessing.
- XGBoost provides complementary decision boundaries and strong regularized tree boosting.
- Blending both usually improves stability and leaderboard performance versus a single model.

## OOF and Blending Strategy

### OOF (Out-of-Fold) predictions

For each fold:
- Train on K-1 folds
- Predict on held-out fold
- Store predictions in OOF vectors

This gives unbiased validation predictions across the full training set and allows reliable model comparison.

### Blend optimization

After CV:
- Compute OOF AUC for CatBoost and XGBoost separately
- Search blend weights on OOF predictions (`w_cat`, `w_xgb`) with `w_xgb = 1 - w_cat`
- Select weights that maximize OOF ROC-AUC

Final test predictions:
- Average each model’s test predictions across folds
- Apply optimized blend weights

## Repository Structure

- `train_cat_xgb_ensemble.py`: main training + inference script
- `requirements.txt`: dependencies
- `.gitignore`: local/runtime artifacts to ignore
- `README.md`: documentation

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

From inside this project folder:

```bash
python train_cat_xgb_ensemble.py \
  --train-path ../train.csv \
  --test-path ../test.csv \
  --sample-submission-path ../sample_submission.csv \
  --submission-out submission.csv \
  --report-out report.json
```

## Outputs

1. `submission.csv`
- Kaggle-ready file with required schema:
  - `id`
  - `Heart Disease` (probability)

2. `report.json`
- Fold-level AUC metrics
- OOF AUC for each base model
- Best blend weights and blended OOF AUC
- Output paths and run metadata

## Reproducibility Notes

- Uses explicit random seed argument (`--seed`)
- Uses stratified CV for consistent class balance per fold
- OOF-based blending reduces leaderboard overfitting risk versus holdout-only blending

## Recommended Next Improvements

- Add Optuna tuning for CatBoost/XGBoost parameters
- Add multi-seed ensembling per model
- Add model calibration experiments (if needed for downstream thresholding)
- Add CI checks and experiment tracking (e.g., MLflow/W&B)

## License / Usage

This project is intended for educational and competition workflows. Please follow Kaggle competition rules and terms for submissions and dataset usage.
