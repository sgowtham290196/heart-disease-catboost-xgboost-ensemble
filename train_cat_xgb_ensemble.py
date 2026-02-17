#!/usr/bin/env python3
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CatBoost + XGBoost ensemble and generate Kaggle submission")
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--sample-submission-path", required=True)
    parser.add_argument("--target-col", default="Heart Disease")
    parser.add_argument("--id-col", default="id")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-step", type=float, default=0.02)
    parser.add_argument("--submission-out", default="submission.csv")
    parser.add_argument("--report-out", default="report.json")
    return parser.parse_args()


def detect_cat_cols(df: pd.DataFrame) -> list[str]:
    cat_cols: list[str] = []
    for col in df.columns:
        if df[col].dtype == "object":
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique(dropna=False) <= 16:
            cat_cols.append(col)
    return cat_cols


def optimize_blend(y_true: np.ndarray, pred_cat: np.ndarray, pred_xgb: np.ndarray, step: float) -> tuple[dict[str, float], float]:
    best_auc = -1.0
    best_weights = {"catboost": 0.5, "xgboost": 0.5}
    for w_cat in np.arange(0.0, 1.0 + 1e-12, step):
        w_xgb = 1.0 - w_cat
        blended = w_cat * pred_cat + w_xgb * pred_xgb
        auc = roc_auc_score(y_true, blended)
        if auc > best_auc:
            best_auc = float(auc)
            best_weights = {"catboost": float(w_cat), "xgboost": float(w_xgb)}
    return best_weights, best_auc


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    sample_sub = pd.read_csv(args.sample_submission_path)

    y_series = train_df[args.target_col]
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_series.astype(int).to_numpy()
    else:
        y = y_series.map({"Absence": 0, "Presence": 1}).astype(int).to_numpy()

    feature_cols = [c for c in train_df.columns if c not in [args.target_col, args.id_col]]
    x = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()
    cat_cols = detect_cat_cols(x)

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    oof_cat = np.zeros(len(x), dtype=np.float64)
    oof_xgb = np.zeros(len(x), dtype=np.float64)
    pred_cat = np.zeros(len(x_test), dtype=np.float64)
    pred_xgb = np.zeros(len(x_test), dtype=np.float64)

    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(x, y), start=1):
        x_tr, y_tr = x.iloc[tr_idx], y[tr_idx]
        x_va, y_va = x.iloc[va_idx], y[va_idx]

        cat_model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=args.seed + fold,
            verbose=False,
        )
        cat_model.fit(
            x_tr,
            y_tr,
            eval_set=(x_va, y_va),
            cat_features=cat_cols,
            use_best_model=True,
            early_stopping_rounds=150,
        )

        xgb_model = XGBClassifier(
            objective="binary:logistic",
            n_estimators=1800,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=120,
            random_state=args.seed + fold,
            n_jobs=-1,
        )
        xgb_model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)

        va_cat = cat_model.predict_proba(x_va)[:, 1]
        va_xgb = xgb_model.predict_proba(x_va)[:, 1]

        te_cat = cat_model.predict_proba(x_test)[:, 1]
        te_xgb = xgb_model.predict_proba(x_test)[:, 1]

        oof_cat[va_idx] = va_cat
        oof_xgb[va_idx] = va_xgb
        pred_cat += te_cat / args.n_splits
        pred_xgb += te_xgb / args.n_splits

        fold_metrics.append(
            {
                "fold": fold,
                "catboost_auc": float(roc_auc_score(y_va, va_cat)),
                "xgboost_auc": float(roc_auc_score(y_va, va_xgb)),
            }
        )

    auc_cat = float(roc_auc_score(y, oof_cat))
    auc_xgb = float(roc_auc_score(y, oof_xgb))

    blend_weights, blend_auc = optimize_blend(y, oof_cat, oof_xgb, args.blend_step)
    pred_final = blend_weights["catboost"] * pred_cat + blend_weights["xgboost"] * pred_xgb

    submission = sample_sub.copy()
    submission[args.id_col] = test_df[args.id_col]
    submission[args.target_col] = pred_final
    submission.to_csv(args.submission_out, index=False)

    report = {
        "n_splits": args.n_splits,
        "seed": args.seed,
        "features": len(feature_cols),
        "catboost_categorical_columns": cat_cols,
        "oof_auc": {
            "catboost": auc_cat,
            "xgboost": auc_xgb,
            "blend": blend_auc,
        },
        "blend_weights": blend_weights,
        "fold_metrics": fold_metrics,
        "outputs": {
            "submission": args.submission_out,
            "report": args.report_out,
        },
    }

    with open(args.report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved submission:", args.submission_out)
    print("Saved report:", args.report_out)


if __name__ == "__main__":
    main()
