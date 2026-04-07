import ast
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soccerdata as sd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from rich import print
from scipy.stats import poisson
from sklearn.calibration import calibration_curve

from const import TEAMS_DATA_DIR, TEAMS_PROJECTION_DIR, leagues, season, teams
from utils import ci, get_gameweek


def create_model_data(df):
    home_df = pd.DataFrame(
        {
            "date": df["date"].dt.tz_localize(None),
            "team": df["home_team"],
            "opponent": df["away_team"],
            "goals": df["home_score"],
            "home": 1,
        }
    )

    away_df = pd.DataFrame(
        {
            "date": df["date"].dt.tz_localize(None),
            "team": df["away_team"],
            "opponent": df["home_team"],
            "goals": df["away_score"],
            "home": 0,
        }
    )

    return pd.concat([home_df, away_df])


def fit_model(df, T, lam=0.01):
    model_df = df.copy()
    model_df["days_ago"] = (T - model_df["date"]).dt.days
    model_df["weight"] = np.exp(-lam * model_df["days_ago"])

    model = smf.glm(
        formula="goals ~ home + team + opponent",
        data=model_df,
        family=sm.families.Poisson(),
        freq_weights=model_df["weight"],
    ).fit()

    return model


def _predict(
    model: statsmodels.genmod.generalized_linear_model.GLMResultsWrapper,
    home: str,
    away: str,
) -> tuple[str, str]:
    home_mean = model.predict(
        pd.DataFrame(
            data={"team": home, "opponent": away, "home": 1},
            index=[1],
        )
    ).values[0]

    away_mean = model.predict(
        pd.DataFrame(
            data={"team": away, "opponent": home, "home": 0},
            index=[1],
        )
    ).values[0]

    return home_mean, away_mean


def process_predict(proba: list, home: str, away: str) -> tuple[dict, dict]:

    long_info, short_info = {}, {}

    score_matrix = np.outer(proba[0], proba[1])
    np.round(score_matrix)

    home_prob = np.round(np.sum(np.tril(score_matrix, -1)), 4)
    draw_prob = np.round(np.trace(score_matrix), 4)
    away_prob = np.round(np.sum(np.triu(score_matrix, 1)), 4)

    outcomes = np.array([home_prob, draw_prob, away_prob])
    labels = np.array([home, "Draw", away])
    prediction = labels[np.argmax(outcomes)]

    long_info["Predicted Outcome"] = prediction

    print(f"{prediction} ({np.max(outcomes)})")
    short_info["Outcome"] = f"{prediction} ({np.max(outcomes)})"

    flat_idx = np.argmax(score_matrix)
    row, col = np.unravel_index(flat_idx, score_matrix.shape)

    long_info["Home Probability"] = home_prob
    long_info["Away Probability"] = away_prob
    long_info["Draw Probability"] = draw_prob
    long_info["Most likely score"] = f"{row}-{col}"
    long_info["Most likely score probability"] = float(
        np.round(score_matrix[row, col], 4)
    )

    rows, cols = np.indices(score_matrix.shape)
    total_goals_grid = rows + cols

    thresholds = [2.5, 3.5]

    for t in thresholds:
        over_prob = score_matrix[total_goals_grid > t].sum()
        under_prob = 1 - over_prob

        long_info[f"Over {t}"] = float(np.round(over_prob, 4))
        long_info[f"Under {t}"] = float(np.round(under_prob, 4))

        if over_prob > under_prob:
            short_info[t] = f"Over ({over_prob:.4f})"
            print(f"Over {t} ({over_prob:.4f})")
        else:
            short_info[t] = f"Under ({under_prob:.4f})"
            print(f"Under {t} ({under_prob:.4f})")

    print("=" * 50)

    return long_info, short_info


# add a field that records the date of the most recent match included in calculation for each leagues
# use this date to fetch matches after this one


def calc_accuracy():
    sofascore = sd.Sofascore(leagues=leagues, seasons=season)
    results = sofascore.read_schedule(force_cache=True)
    results["date"] = results["date"].dt.tz_localize(None).dt.date

    df = pd.read_csv("prediction/output/accuracy.csv", index_col=0)

    for league in leagues:
        league_code, league_name = league.split("-")
        (
            latest_date,
            total_matches,
            correct_outcome,
            correct_outcome_min,
            correct_outcome_max,
            correct_outcome_avg,
            correct_2_5,
            correct_2_5_min,
            correct_2_5_max,
            correct_2_5_avg,
            correct_3_5,
            correct_3_5_min,
            correct_3_5_max,
            correct_3_5_avg,
        ) = df.loc[league].values

        latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()

        result_df = results.query("league == @league and date > @latest_date").dropna()

        if result_df.empty:
            print(f"No results fetched for {league_name}.")
            continue

        curr_correct_outcome, curr_correct_2_5, curr_correct_3_5, curr_total = (
            0,
            0,
            0,
            0,
        )

        curr_correct_outcome_probs, curr_correct_2_5_probs, curr_correct_3_5_probs = (
            [],
            [],
            [],
        )

        gw = result_df.iloc[0]["week"]
        prediction = read_predictions(league_code, league_name, gw)

        for index, row in result_df.iterrows():
            if row["week"] != gw:
                gw = row["week"]
                prediction = read_predictions(league_code, league_name, gw)
            if not prediction:
                print(f"No predictions found for {league_name} on gw{row['week']}.")
                continue
            match_prediction = prediction[index[2].split(" ", maxsplit=1)[1]]
            home_score = row["home_score"]
            away_score = row["away_score"]
            total_goals = home_score + away_score

            if home_score > away_score:
                outcome = row["home_team"]
                outcome_side = "Home"
            elif away_score > home_score:
                outcome = row["away_team"]
                outcome_side = "Away"
            else:
                outcome = "Draw"
                outcome_side = outcome

            if outcome == match_prediction["Predicted Outcome"]:
                curr_correct_outcome += 1
                curr_correct_outcome_probs.append(
                    match_prediction[f"{outcome_side} Probability"]
                )

            if (
                match_prediction["Over 2.5"] > match_prediction["Under 2.5"]
                and total_goals > 2.5
            ):
                curr_correct_2_5 += 1
                curr_correct_2_5_probs.append(match_prediction["Over 2.5"])
            elif (
                match_prediction["Over 2.5"] < match_prediction["Under 2.5"]
                and total_goals < 2.5
            ):
                curr_correct_2_5 += 1
                curr_correct_2_5_probs.append(match_prediction["Under 2.5"])

            if (
                match_prediction["Over 3.5"] > match_prediction["Under 3.5"]
                and total_goals > 3.5
            ):
                curr_correct_3_5 += 1
                curr_correct_3_5_probs.append(match_prediction["Over 3.5"])
            elif (
                match_prediction["Over 3.5"] < match_prediction["Under 3.5"]
                and total_goals < 3.5
            ):
                curr_correct_3_5 += 1
                curr_correct_3_5_probs.append(match_prediction["Under 3.5"])

            curr_total += 1

        curr_correct_outcome_probs = [
            round(x * 100, 2) for x in curr_correct_outcome_probs
        ]
        curr_correct_2_5_probs = [round(x * 100, 2) for x in curr_correct_2_5_probs]
        curr_correct_3_5_probs = [round(x * 100, 2) for x in curr_correct_3_5_probs]

        league_df = df.loc[league].copy()
        league_df["latest_date"] = result_df.iloc[-1]["date"]
        league_df["total_matches"] = total_matches + curr_total

        league_df["correct_outcome"] = (
            ((correct_outcome / 100) * total_matches + curr_correct_outcome)
            / (total_matches + curr_total)
            * 100
        )
        league_df["correct_outcome_min"] = min(
            correct_outcome_min, min(curr_correct_outcome_probs)
        )
        league_df["correct_outcome_max"] = max(
            correct_outcome_max, max(curr_correct_outcome_probs)
        )
        league_df["correct_outcome_avg"] = (
            correct_outcome_avg * (correct_outcome / 100 * total_matches)
            + sum(curr_correct_outcome_probs)
        ) / (correct_outcome / 100 * total_matches + curr_correct_outcome)

        league_df["correct_2_5"] = (
            ((correct_2_5 / 100) * total_matches + curr_correct_2_5)
            / (total_matches + curr_total)
            * 100
        )
        league_df["correct_2_5_min"] = min(correct_2_5_min, min(curr_correct_2_5_probs))
        league_df["correct_2_5_max"] = max(correct_2_5_max, max(curr_correct_2_5_probs))
        league_df["correct_2_5_avg"] = (
            correct_2_5_avg * (correct_2_5 / 100 * total_matches)
            + sum(curr_correct_2_5_probs)
        ) / (correct_2_5 / 100 * total_matches + curr_correct_2_5)

        league_df["correct_3_5"] = (
            ((correct_3_5 / 100) * total_matches + curr_correct_3_5)
            / (total_matches + curr_total)
            * 100
        )
        league_df["correct_3_5_min"] = min(correct_3_5_min, min(curr_correct_3_5_probs))
        league_df["correct_3_5_max"] = max(correct_3_5_max, max(curr_correct_3_5_probs))
        league_df["correct_3_5_avg"] = (
            correct_3_5_avg * (correct_3_5 / 100 * total_matches)
            + sum(curr_correct_3_5_probs)
        ) / (correct_3_5 / 100 * total_matches + curr_correct_3_5)

        df.loc[league] = league_df.values

    df = df.round(2)
    df.sort_values(
        ["correct_2_5", "correct_3_5", "correct_outcome"], ascending=False, inplace=True
    )
    df.to_csv("prediction/output/accuracy.csv")
    print(df)


def init_calc_accuracy():
    sofascore = sd.Sofascore(leagues=leagues, seasons=season)
    results = sofascore.read_schedule(force_cache=True)
    # gameweeks = get_gameweek()
    gameweeks = {
        "ENG-Premier League": "28",
        "ESP-La Liga": "26",
        "FRA-Ligue 1": "24",
        "GER-Bundesliga": "24",
        "ITA-Serie A": "27",
    }
    res = []

    for league in leagues:
        correct_outcome, correct_2_5, correct_3_5 = 0, 0, 0
        correct_outcome_probs, correct_2_5_probs, correct_3_5_probs = [], [], []
        league_code, league_name = league.split("-")
        gw = int(gameweeks[league])

        result_df = results.query("league == @league and week == @gw").dropna()

        if result_df.empty:
            gw -= 1
            result_df = results.query("league == @league and week == @gw").dropna()

        with open(
            f"prediction/data/{league_code}/{league_code}_gw{gw}_long.json",
            "r",
            encoding="utf-8",
        ) as file:
            prediction = json.load(file)

        for match, match_info in prediction.items():
            match_result = result_df.loc[
                result_df.index.get_level_values("game").str.contains(match)
            ]
            if not match_result.empty:
                home_score = match_result["home_score"].values[0]
                away_score = match_result["away_score"].values[0]
                total_goals = home_score + away_score

                if home_score > away_score:
                    outcome = match_result["home_team"].values[0]
                    outcome_side = "Home"
                elif away_score > home_score:
                    outcome = match_result["away_team"].values[0]
                    outcome_side = "Away"
                else:
                    outcome = "Draw"
                    outcome_side = outcome

                if outcome == match_info["Predicted Outcome"]:
                    correct_outcome += 1
                    correct_outcome_probs.append(
                        match_info[f"{outcome_side} Probability"]
                    )

                if (
                    match_info["Over 2.5"] > match_info["Under 2.5"]
                    and total_goals > 2.5
                ):
                    correct_2_5 += 1
                    correct_2_5_probs.append(match_info["Over 2.5"])
                elif (
                    match_info["Over 2.5"] < match_info["Under 2.5"]
                    and total_goals < 2.5
                ):
                    correct_2_5 += 1
                    correct_2_5_probs.append(match_info["Under 2.5"])

                if (
                    match_info["Over 3.5"] > match_info["Under 3.5"]
                    and total_goals > 3.5
                ):
                    correct_3_5 += 1
                    correct_3_5_probs.append(match_info["Over 3.5"])
                elif (
                    match_info["Over 3.5"] < match_info["Under 3.5"]
                    and total_goals < 3.5
                ):
                    correct_3_5 += 1
                    correct_3_5_probs.append(match_info["Under 3.5"])

        total_matches = len(result_df)
        correct_outcome_rate = round(((correct_outcome / total_matches) * 100), 2)
        correct_2_5_rate = round(((correct_2_5 / total_matches) * 100), 2)
        correct_3_5_rate = round(((correct_3_5 / total_matches) * 100), 2)

        correct_outcome_probs = [round(x * 100, 2) for x in correct_outcome_probs]
        correct_2_5_probs = [round(x * 100, 2) for x in correct_2_5_probs]
        correct_3_5_probs = [round(x * 100, 2) for x in correct_3_5_probs]

        res.append(
            {
                "league": league,
                "latest_date": result_df.iloc[-1]["date"].strftime("%Y-%m-%d"),
                "total_matches": total_matches,
                "correct_outcome": round(correct_outcome_rate, 2),
                "correct_outcome_min": min(correct_outcome_probs),
                "correct_outcome_max": max(correct_outcome_probs),
                "correct_outcome_avg": round(
                    sum(correct_outcome_probs) / len(correct_outcome_probs), 2
                ),
                "correct_2_5": round(correct_2_5_rate, 2),
                "correct_2_5_min": min(correct_2_5_probs),
                "correct_2_5_max": max(correct_2_5_probs),
                "correct_2_5_avg": round(
                    sum(correct_2_5_probs) / len(correct_2_5_probs), 2
                ),
                "correct_3_5": round(correct_3_5_rate, 2),
                "correct_3_5_min": min(correct_3_5_probs),
                "correct_3_5_max": max(correct_3_5_probs),
                "correct_3_5_avg": round(
                    sum(correct_3_5_probs) / len(correct_3_5_probs), 2
                ),
            }
        )
    df = pd.DataFrame(res)
    df.sort_values(
        ["correct_2_5", "correct_3_5", "correct_outcome"], ascending=False, inplace=True
    )
    df.to_csv("prediction/output/accuracy.csv", index=False)
    print(df)


def read_predictions(league_code, league_name, gw) -> dict[str, dict] | None:
    try:
        with open(
            f"prediction/data/{league_code}/{league_code}_gw{gw}_long.json",
            "r",
            encoding="utf-8",
        ) as file:
            prediction = json.load(file)
            return prediction
    except FileNotFoundError as e:
        print(f"{e}: No prediction found for {league_name} gw{gw}.")
        return None


# Add latest_date field
# Filter matches after latest_date


def init_log_results():
    sofascore = sd.Sofascore(leagues=leagues, seasons=season)
    results = sofascore.read_schedule(force_cache=True)
    gameweeks = {
        "ENG-Premier League": "28",
        "ESP-La Liga": "26",
        "FRA-Ligue 1": "24",
        "GER-Bundesliga": "24",
        "ITA-Serie A": "27",
    }
    res = []

    for league in leagues:
        league_code, league_name = league.split("-")
        gw = int(gameweeks[league])

        result_df = results.query("league == @league and week == @gw").dropna()

        if result_df.empty:
            print(f"Error: No results fetched for {league} on {gw}")
            continue

        try:
            with open(
                f"prediction/data/{league_code}/{league_code}_gw{gw}_long.json",
                "r",
                encoding="utf-8",
            ) as file:
                prediction = json.load(file)
        except FileNotFoundError as e:
            print(f"{e}: No prediction found for {league_name} gw {gw}.")
            continue

        for match, match_info in prediction.items():
            match_result = result_df.loc[
                result_df.index.get_level_values("game").str.contains(match)
            ]
            if not match_result.empty:
                home_score = match_result["home_score"].values[0]
                away_score = match_result["away_score"].values[0]
                total_goals = home_score + away_score

                home_outcome = 1 if home_score > away_score else 0
                away_outcome = 1 if away_score > home_score else 0
                draw_outcome = 1 if home_score == away_score else 0

                u_2_5_outcome = 1 if total_goals < 2.5 else 0
                o_2_5_outcome = 1 if total_goals > 2.5 else 0
                u_3_5_outcome = 1 if total_goals < 3.5 else 0
                o_3_5_outcome = 1 if total_goals > 3.5 else 0

                latest_date = pd.to_datetime(match_result["date"].values[0]).date()

                res.extend(
                    [
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Home Probability"],
                            "outcome": home_outcome,
                            "market_type": "moneyline",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Away Probability"],
                            "outcome": away_outcome,
                            "market_type": "moneyline",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Draw Probability"],
                            "outcome": draw_outcome,
                            "market_type": "moneyline",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Over 2.5"],
                            "outcome": o_2_5_outcome,
                            "market_type": "2_5",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Under 2.5"],
                            "outcome": u_2_5_outcome,
                            "market_type": "2_5",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Over 3.5"],
                            "outcome": o_3_5_outcome,
                            "market_type": "3_5",
                        },
                        {
                            "latest_date": latest_date,
                            "predicted_prob": match_info["Under 3.5"],
                            "outcome": u_3_5_outcome,
                            "market_type": "3_5",
                        },
                    ]
                )

    df = pd.DataFrame(res)
    df.sort_values(by="latest_date", ascending=False, inplace=True)
    df.to_csv("prediction/output/log.csv", index=False)
    print(df)


def log_results():
    sofascore = sd.Sofascore(leagues=leagues, seasons=season)
    results = sofascore.read_schedule(force_cache=True)
    results["date"] = results["date"].dt.tz_localize(None).dt.date

    prev_log_df = pd.read_csv("prediction/output/log.csv")
    latest_date = prev_log_df["latest_date"].max()
    latest_date = datetime.strptime(latest_date, "%Y-%m-%d").date()

    res = []

    for league in leagues:
        league_code, league_name = league.split("-")

        result_df = results.query("league == @league and date > @latest_date").dropna()

        if result_df.empty:
            print(f"Error: No results fetched for {league}")
            continue

        gw = result_df.iloc[0]["week"]
        prediction = read_predictions(league_code, league_name, gw)

        if not prediction:
            print(f"No predictions found for {league_name} on gw{gw}.")
            continue

        for index, row in result_df.iterrows():
            if row["week"] != gw:
                gw = row["week"]
                prediction = read_predictions(league_code, league_name, gw)
            if not prediction:
                print(f"No predictions found for {league_name} on gw{row['week']}.")
                break
            match_prediction = prediction[index[2].split(" ", maxsplit=1)[1]]

            home_score = row["home_score"]
            away_score = row["away_score"]
            total_goals = home_score + away_score

            home_outcome = 1 if home_score > away_score else 0
            away_outcome = 1 if away_score > home_score else 0
            draw_outcome = 1 if home_score == away_score else 0

            u_2_5_outcome = 1 if total_goals < 2.5 else 0
            o_2_5_outcome = 1 if total_goals > 2.5 else 0
            u_3_5_outcome = 1 if total_goals < 3.5 else 0
            o_3_5_outcome = 1 if total_goals > 3.5 else 0

            res.extend(
                [
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Home Probability"],
                        "outcome": home_outcome,
                        "market_type": "moneyline",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Away Probability"],
                        "outcome": away_outcome,
                        "market_type": "moneyline",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Draw Probability"],
                        "outcome": draw_outcome,
                        "market_type": "moneyline",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Over 2.5"],
                        "outcome": o_2_5_outcome,
                        "market_type": "2_5",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Under 2.5"],
                        "outcome": u_2_5_outcome,
                        "market_type": "2_5",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Over 3.5"],
                        "outcome": o_3_5_outcome,
                        "market_type": "3_5",
                    },
                    {
                        "latest_date": row["date"],
                        "predicted_prob": match_prediction["Under 3.5"],
                        "outcome": u_3_5_outcome,
                        "market_type": "3_5",
                    },
                ]
            )

    prev_log_df = pd.read_csv("prediction/output/log.csv")
    df = pd.concat([pd.DataFrame(res), prev_log_df])
    df.to_csv("prediction/output/log.csv", index=False)
    print(df)


def plot_calibration():
    df = pd.read_csv("prediction/output/log.csv")
    markets = df["market_type"].unique()
    n_bins = 5
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for market in markets:
        market_df = df[df["market_type"] == market]
        y_true = market_df["outcome"]
        y_prob = market_df["predicted_prob"]

        market_fop, market_mpv = calibration_curve(y_true, y_prob, n_bins=n_bins)
        plt.plot(market_mpv, market_fop, marker=".", label=market)

        counts, _ = np.histogram(y_prob, bins=bin_edges)
        print(f"\n{market} — Sample size per bin:")
        print(f"{'Bin Range':>20} | {'Count':>6} |")
        print("-" * 45)
        for j, (edge_low, edge_high, count) in enumerate(
            zip(bin_edges[:-1], bin_edges[1:], counts)
        ):
            print(f"  {edge_low:.2f} - {edge_high:.2f}        | {count:>6} |")

    plt.plot(
        [0, 1], [0, 1], linestyle="--", color="black", label="Perfectly calibrated"
    )
    fop, mpb = calibration_curve(df["outcome"], df["predicted_prob"], n_bins=5)
    plt.plot(mpb, fop, marker=".", label="Overall")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(f"Calibration Curves as at {df['latest_date'].max()}")

    counts, _ = np.histogram(df["predicted_prob"], bins=bin_edges)
    print("\nOverall — Sample size per bin:")
    print(f"{'Bin Range':>20} | {'Count':>6} |")
    print("-" * 45)
    for j, (edge_low, edge_high, count) in enumerate(
        zip(bin_edges[:-1], bin_edges[1:], counts)
    ):
        print(f"  {edge_low:.2f} - {edge_high:.2f}        | {count:>6} |")
    plt.legend()
    plt.savefig("prediction/output/calibration_curves.png")
    plt.show()


def calc_brier():
    df = pd.read_csv("prediction/output/log.csv")
    df["squared_error"] = (df["predicted_prob"] - df["outcome"]) ** 2
    results = df.groupby("market_type")["squared_error"].mean()
    results.to_json("prediction/output/brier.json", indent=4)
    print(results)


def predict():
    max_goals = 6
    gameweeks = get_gameweek()
    for league in leagues:
        league_code, league_name = league.split("-")

        ss = sd.Sofascore(leagues=league, seasons=season, proxy="tor")
        hist = ss.read_schedule(force_cache=True)
        gw = gameweeks[league]
        T = hist.loc[hist["week"] == int(gw), "date"].values[0]
        hist.dropna(inplace=True)
        hist.reset_index(drop=True, inplace=True)
        cols = ["date", "home_team", "away_team", "home_score", "away_score"]
        league_df = create_model_data(hist[cols].copy())
        model = fit_model(league_df, T)

        fixtures_df = pd.read_csv(TEAMS_DATA_DIR / f"{league_code}_fixtures.csv")
        fixtures_df.set_index("team", inplace=True)

        if league_code == "ENG":
            pl_next, pl_next5 = [], []
            for team in teams[league_name]:
                expected_gf, expected_ga = 0.0, 0.0
                opponents = []
                for pl_gw in range(int(gw), int(gw) + 5):
                    val = fixtures_df.at[team, str(pl_gw)]
                    if pd.isna(val):
                        print(f"Blank GW for {team} on gw {pl_gw}")
                        continue
                    curr_fixtures = ast.literal_eval(fixtures_df.at[team, str(pl_gw)])
                    for curr_fixture in curr_fixtures:
                        opponent, side = curr_fixture.split("-")
                        opponents.append(f"{opponent}-{side}")
                        if side == "H":
                            team_mean, opponent_mean = _predict(model, team, opponent)
                        else:
                            opponent_mean, team_mean = _predict(model, opponent, team)

                        expected_gf += team_mean
                        expected_ga += opponent_mean

                pl_next5.append(
                    {
                        "team": team,
                        "opponents": opponents,
                        "expected_gf": round(expected_gf, 2),
                        "expected_ga": round(expected_ga, 2),
                    }
                )

        long, short = {}, {}
        for team in teams[league_name]:
            curr_fixtures = fixtures_df.at[team, gw]
            if pd.isna(curr_fixtures):
                print("Blank")
                continue
            curr_fixtures = ast.literal_eval(fixtures_df.at[team, gw])
            for curr_fixture in curr_fixtures:
                opponent, side = curr_fixture.split("-")
                team_mean, opponent_mean = _predict(model, team, opponent)

                if league_code == "ENG":
                    if side == "A":
                        opponent_mean, team_mean = _predict(model, opponent, team)
                    pl_next.append(
                        {
                            "team": team,
                            "opponents": opponent,
                            "expected_gf": round(team_mean, 2),
                            "expected_ga": round(opponent_mean, 2),
                        }
                    )

                if side == "A":
                    continue

                proba = [
                    [poisson.pmf(i, mean) for i in range(0, max_goals)]
                    for mean in [team_mean, opponent_mean]
                ]

                print(
                    f"Prediction for the game between {team} ({team_mean:.2f}) and {opponent} ({opponent_mean:.2f}))"
                )

                long_info, short_info = process_predict(proba, team, opponent)

                long[f"{team}-{opponent}"] = long_info
                short[f"{team}-{opponent}"] = short_info

        with open(
            f"prediction/data/{league_code}/{league_code}_gw{gw}_long.json", "w"
        ) as file:
            json.dump(long, file, indent=4)

        with open(
            f"prediction/data/{league_code}/{league_code}_gw{gw}_short.json", "w"
        ) as file:
            json.dump(short, file, indent=4)

        if league_code == "ENG":
            pl_next5_df = pd.DataFrame(pl_next5)
            pl_next5_df.sort_values(by=["expected_gf"], ascending=False, inplace=True)
            pl_next5_df.to_csv(TEAMS_PROJECTION_DIR / "next5forecast.csv", index=False)

            pl_df = pd.DataFrame(pl_next)
            pl_df.sort_values(by=["expected_gf"], ascending=False, inplace=True)
            pl_df.to_csv(TEAMS_PROJECTION_DIR / "nextforecast.csv", index=False)
