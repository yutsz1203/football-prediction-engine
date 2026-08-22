import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from statsmodels.stats.outliers_influence import variance_inflation_factor

from const import MODEL_RESULTS_DIR, PLAYERS_RESULTS_DIR


def save_ols_results(model, pos: str) -> None:
    MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.summary2().tables[1].to_csv(MODEL_RESULTS_DIR / f"{pos}_coefs.csv")
    (MODEL_RESULTS_DIR / f"{pos}_summary.txt").write_text(model.summary().as_text())


def fit_ols(df: pd.DataFrame, features: list[str], pos: str) -> None:
    X = df[features]
    X = sm.add_constant(X)
    y = df["total_points"]
    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["element"]})
    print(model.summary())
    save_ols_results(model, pos)


def check_vif(df: pd.DataFrame):
    X = df.filter(like="roll_")
    vifs = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns,
    )
    return vifs.sort_values(ascending=False)


def select_features_by_vif(df: pd.DataFrame, threshold: float = 20.0) -> list[str]:
    features = list(df.filter(like="roll_").columns)
    remove = [
        "roll_yellow_cards_p90",
        "roll_transfers_balance",
        "roll_selected",
        "roll_bps_p90",
        "roll_value",
    ]
    features = [x for x in features if x not in remove]
    while True:
        X = df[features].values
        vifs = pd.Series(
            [variance_inflation_factor(X, i) for i in range(len(features))],
            index=features,
        )
        worst = vifs.idxmax()
        if vifs[worst] > threshold and worst != "roll_minutes":
            print(f"  Dropping {worst!r} (VIF={vifs[worst]:.1f})")
            features.remove(worst)
        else:
            break
    return features


def expanding_window_eval(df, candidate_features, pos_label, start_gw=15, end_gw=39):
    """
    Expanding window evaluation using the splits approach:
    train on all GWs before test_gw, test on test_gw.
    """
    results = []
    feature_tracker = {f: 0 for f in candidate_features}

    # Build splits
    df = df.sort_values("round")
    splits = []
    for test_gw in range(start_gw, end_gw):
        train = df[df["round"] < test_gw]
        test = df[df["round"] == test_gw]
        if len(train) >= 50 and len(test) >= 3:
            splits.append((train, test, test_gw))

    for train, test, test_gw in splits:

        # --- Lasso feature selection on training data only ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train[candidate_features])

        lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, train["total_points"])

        selected = [f for f, c in zip(candidate_features, lasso.coef_) if c != 0]

        for f in selected:
            feature_tracker[f] += 1

        if len(selected) == 0:
            selected = candidate_features

        # --- OLS refit on selected features ---
        X_train = sm.add_constant(train[selected])
        X_test = sm.add_constant(test[selected])
        model = sm.OLS(train["total_points"], X_train).fit()
        preds = model.predict(X_test)

        # --- Baselines ---
        actuals = test["total_points"]
        baseline_mean_mae = (actuals - train["total_points"].mean()).abs().mean()

        # --- Metrics ---
        mae = (actuals - preds).abs().mean()
        rmse = np.sqrt(((actuals - preds) ** 2).mean())

        results.append(
            {
                "gw": test_gw,
                "mae": mae,
                "rmse": rmse,
                "baseline_mean_mae": baseline_mean_mae,
                "n_train": len(train),
                "n_test": len(test),
                "n_features": len(selected),
                "features": selected,
            }
        )

    results_df = pd.DataFrame(results)

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Position: {pos_label}")
    print(f"{'='*60}")
    print(f"GWs evaluated:     {len(results_df)}")
    print(f"Mean MAE:          {results_df['mae'].mean():.3f}")
    print(f"Mean Baseline MAE: {results_df['baseline_mean_mae'].mean():.3f}")
    diff = results_df["baseline_mean_mae"].mean() - results_df["mae"].mean()
    print(f"Improvement:       {diff:+.3f}")
    print(f"Mean RMSE:         {results_df['rmse'].mean():.3f}")

    print(f"\nFeature selection frequency ({len(splits)} folds):")
    freq = pd.Series(feature_tracker).sort_values(ascending=False)
    total_folds = len(splits)
    for f, count in freq.items():
        pct = count / total_folds * 100
        bar = "█" * int(pct // 5)
        print(f"  {f:45s} {count:3d}/{total_folds}  ({pct:5.1f}%)  {bar}")

    freq /= total_folds
    final_features = freq[freq > 0.5].index.tolist()

    return results_df, final_features


if __name__ == "__main__":
    file_path = PLAYERS_RESULTS_DIR / "gameweek_level"
    pos = ["gk", "def", "mid", "fwd"]

    for p in pos:
        print(p)
        df = pd.read_csv(f"{file_path}/{p}_2526.csv")
        features = select_features_by_vif(df)

        results, final_features = expanding_window_eval(
            df, features, pos_label=p, start_gw=15, end_gw=39
        )

        print(f"\nTop features for {p}: {final_features}")
        fit_ols(df, final_features, p)
