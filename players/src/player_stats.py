# url: https://fantasy.premierleague.com/api/element-summary/{player-id}/
# or try this: https://fantasy.premierleague.com/api/event/1/live/
import asyncio
import os
import sys
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from const import (  # noqa : E402
    PLAYERS_RESULTS_DIR,
    element_type_map,
    official_base_url,
    official_element_summary,
)
from players.src.schemas import PlayerStats

ROLLING_N = 6
semaphore = asyncio.Semaphore(10)


def stats_calculation(matches: list[dict[str, Any]]) -> PlayerStats:
    stats = PlayerStats()

    for match in matches:
        if match["minutes"] == 0:
            continue
        stats.points += match["total_points"]
        stats.minutes += match["minutes"]
        stats.goals += match["goals_scored"]
        stats.assists += match["assists"]
        stats.clean_sheets += match["clean_sheets"]
        stats.goals_conceded += match["goals_conceded"]
        stats.own_goals += match["own_goals"]
        stats.penalties_saved += match["penalties_saved"]
        stats.penalties_missed += match["penalties_missed"]
        stats.yellow_cards += match["yellow_cards"]
        stats.red_cards += match["red_cards"]
        stats.saves += match["saves"]
        stats.bonus += match["bonus"]
        stats.bps += match["bps"]
        stats.influence += float(match["influence"])
        stats.creativity += float(match["creativity"])
        stats.threat += float(match["threat"])
        stats.ict_index += float(match["ict_index"])
        stats.tackles += match["tackles"]
        stats.clearances_blocks_interceptions += match[
            "clearances_blocks_interceptions"
        ]
        stats.recoveries += match["recoveries"]
        stats.defensive_contribution += match["defensive_contribution"]
        stats.starts += match["starts"]
        stats.xG += float(match["expected_goals"])
        stats.xA += float(match["expected_assists"])
        stats.xGI += float(match["expected_goal_involvements"])
        stats.xGc += float(match["expected_goals_conceded"])
        stats.transfers_in += match["transfers_in"]
        stats.transfers_out += match["transfers_out"]

    if matches:
        stats.id = matches[-1]["element"]
        stats.value = float(matches[-1]["value"]) / 10
        stats.selected = matches[-1]["selected"]

    float_fields = [
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "xG",
        "xA",
        "xGI",
        "xGc",
        "value",
    ]
    for field in float_fields:
        setattr(stats, field, round(getattr(stats, field), 2))

    return stats


def get_all_players(base_url: str) -> pd.DataFrame:
    """Get the total number of players through the bootstrap-static endpoint."""
    response = requests.get(f"{base_url}/bootstrap-static")
    elements = response.json().get("elements", [])
    if elements:
        return pd.DataFrame(elements)
    else:
        return pd.DataFrame()


async def fetch_player_stats(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    player_id: int,
) -> PlayerStats | None:
    url = f"{official_element_summary}/{player_id}"
    for attempt in range(5):
        try:
            async with semaphore:
                print(f"Fetching stats for player ID: {player_id}")
                async with session.get(url) as response:
                    if response.status == 429:
                        print(f"Rate limited on player {player_id}, retrying in 0.1s")
                        await asyncio.sleep(0.1)
                        continue
                    content = await response.json()
                    matches = content.get("history", [])
                    if matches:
                        return stats_calculation(matches)
                    return None
        except Exception as e:
            print(f"Error fetching player {player_id}: {e}")
            await asyncio.sleep(2**attempt)
    return None


async def season_stats(total_players: int, players_df: pd.DataFrame) -> None:
    async with aiohttp.ClientSession() as session:
        player_tasks = [
            fetch_player_stats(session, semaphore, player_id)
            for player_id in range(1, total_players + 1)
        ]
        player_stats = await asyncio.gather(*player_tasks)
        all_player_stats = [stats for stats in player_stats if stats is not None]
        df = pd.DataFrame([stats.__dict__ for stats in all_player_stats])
        id_to_name = players_df.set_index(players_df["id"].astype(int))["web_name"]

        df["name"] = df["id"].map(id_to_name)
        df["pos"] = players_df["element_type"].map(element_type_map)
        cols = ["name", "pos", "value"]
        df = df[cols + [col for col in df.columns if col not in cols]]
        df.drop(columns=["id"], inplace=True)
        file_path = PLAYERS_RESULTS_DIR / "players_seasonstats_2526.csv"
        df.to_csv(file_path)
        print(f"Player season stats saved to {file_path}")


async def fetch_player_gameweek(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    player_id: int,
) -> list[dict[str, Any]] | None:
    url = f"{official_element_summary}/{player_id}"
    for attempt in range(5):
        try:
            async with semaphore:
                print(f"Fetching stats for player ID: {player_id}")
                async with session.get(url) as response:
                    if response.status == 429:
                        print(f"Rate limited on player {player_id}, retrying in 0.1s")
                        await asyncio.sleep(0.1)
                        continue
                    content = await response.json()
                    matches = content.get("history", [])
                    if matches:
                        return matches
                    return None
        except Exception as e:
            print(f"Error fetching player {player_id}: {e}")
            await asyncio.sleep(2**attempt)
    return None


async def gameweek_stats(total_players: int, players_df: pd.DataFrame) -> pd.DataFrame:
    async with aiohttp.ClientSession() as session:
        player_tasks = [
            fetch_player_gameweek(session, semaphore, player_id)
            for player_id in range(1, total_players + 1)
        ]
        player_stats = await asyncio.gather(*player_tasks)
        all_rows = [row for rows in player_stats if rows is not None for row in rows]
        df = pd.DataFrame(all_rows)
        df = df[
            [
                "element",
                "round",
                "opponent_team",
                "total_points",
                "was_home",
                "minutes",
                "goals_scored",
                "assists",
                "clean_sheets",
                "goals_conceded",
                "yellow_cards",
                "saves",
                "bps",
                "influence",
                "creativity",
                "threat",
                "defensive_contribution",
                "expected_goals",
                "expected_assists",
                "expected_goals_conceded",
                "value",
                "transfers_balance",
                "selected",
            ]
        ]
        id_to_name = players_df.set_index(players_df["id"].astype(int))["web_name"]
        id_to_pos = players_df.set_index(players_df["id"].astype(int))[
            "element_type"
        ].map(element_type_map)
        df["name"] = df["element"].map(id_to_name)
        df["pos"] = df["element"].map(id_to_pos)
        return df


def compute_rolling_features(
    group: pd.DataFrame, rolling_features: list[str], static_features: list[str]
) -> pd.DataFrame:
    group = group.sort_values("round")

    for col in rolling_features:
        rolled_stat = group[col].shift(1).rolling(ROLLING_N, min_periods=3).sum()
        rolled_mins = group["minutes"].shift(1).rolling(ROLLING_N, min_periods=3).sum()

        group[f"roll_{col}_p90"] = rolled_stat / rolled_mins * 90

    for col in static_features:
        group[f"roll_{col}"] = (
            group[col].shift(1).rolling(ROLLING_N, min_periods=3).mean()
        )

    return group


def build_gameweek_level_df(df: pd.DataFrame) -> None:
    df = df[df["minutes"] > 0].copy()
    float_cols = [
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "influence",
        "creativity",
        "threat",
    ]
    df[float_cols] = df[float_cols].astype(float)

    df["goal_overperf"] = df["goals_scored"] - df["expected_goals"]
    df["assist_overperf"] = df["assists"] - df["expected_assists"]
    df["goal_conceded_overperf"] = df["goals_conceded"] - df["expected_goals_conceded"]

    rolling_features = [
        "clean_sheets",
        "yellow_cards",
        "saves",
        "bps",
        "influence",
        "creativity",
        "threat",
        "defensive_contribution",
        "expected_goals",
        "expected_assists",
        "expected_goals_conceded",
        "goal_overperf",
        "assist_overperf",
        "goal_conceded_overperf",
    ]
    static_features = ["minutes", "value", "transfers_balance", "selected"]

    element_col = df["element"]
    df = df.groupby("element", group_keys=False).apply(
        lambda g: compute_rolling_features(g, rolling_features, static_features),
        include_groups=False,
    )
    df["element"] = element_col

    feature_cols = [c for c in df.columns if c.startswith("roll_")]

    context_cols = ["was_home", "opponent_team"]
    target = "total_points"

    model_df = df[
        ["element", "name", "pos", "round", target] + feature_cols + context_cols
    ].copy()
    model_df = model_df.round(2)
    model_df.sort_values(["element", "round"], inplace=True)
    file_path = PLAYERS_RESULTS_DIR / "gameweek_level"

    gk_df = model_df[model_df["pos"] == "GK"].copy()
    gk_df = gk_df[gk_df["roll_minutes"] == 90].copy()
    gk_df.drop(
        columns=[
            "roll_creativity_p90",
            "roll_threat_p90",
            "roll_defensive_contribution_p90",
            "roll_expected_goals_p90",
            "roll_expected_assists_p90",
            "roll_goal_overperf_p90",
            "roll_assist_overperf_p90",
            "roll_influence_p90",
        ],
        inplace=True,
    )
    gk_df.to_csv(f"{file_path}/gk_2526.csv", index=False)

    def_df = model_df[model_df["pos"] == "DEF"].copy()
    def_df = def_df[def_df["roll_minutes"] >= 45].copy()
    def_df.drop(columns=["roll_saves_p90"], inplace=True)
    def_df.to_csv(f"{file_path}/def_2526.csv", index=False)

    mid_df = model_df[model_df["pos"] == "MID"].copy()
    mid_df = mid_df[mid_df["roll_minutes"] >= 45].copy()
    mid_df.drop(columns=["roll_saves_p90"], inplace=True)
    mid_df.to_csv(f"{file_path}/mid_2526.csv", index=False)

    fwd_df = model_df[model_df["pos"] == "FWD"].copy()
    fwd_df = fwd_df[fwd_df["roll_minutes"] >= 45].copy()
    fwd_df.drop(columns=["roll_saves_p90", "roll_influence_p90"], inplace=True)
    fwd_df.to_csv(f"{file_path}/fwd_2526.csv", index=False)
    print(f"Gameweek level stats saved to {file_path}.")


def exploratory_check(high_threshold: float = 0.6, low_threshold: float = 0.1) -> None:
    """Perform exploratory checks on the gameweek-level features to identify highly correlated pairs."""
    file_path = PLAYERS_RESULTS_DIR / "gameweek_level"
    for csv_file in ["gk_2526.csv", "def_2526.csv", "fwd_2526.csv"]:
        df = pd.read_csv(f"{file_path}/{csv_file}")
        roll_cols = [c for c in df.columns if c.startswith("roll_")]
        corr = df[roll_cols].corr(method="pearson")

        pos_label = df["pos"].iloc[0] if "pos" in df.columns else "Unknown"
        print(f"\n{'='*60}")
        print(f"Correlation check — position: {pos_label}  (n={len(df)})")
        print(f"{'='*60}")

        # Collect upper-triangle pairs
        high_pairs: list[tuple[str, str, float]] = []
        low_pairs: list[tuple[str, str, float]] = []

        for i, col_a in enumerate(roll_cols):
            for col_b in roll_cols[i + 1 :]:
                r = corr.loc[col_a, col_b]
                if abs(r) >= high_threshold:
                    high_pairs.append((col_a, col_b, r))
                elif abs(r) <= low_threshold:
                    low_pairs.append((col_a, col_b, r))

        high_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        low_pairs.sort(key=lambda x: abs(x[2]))

        print(
            f"\nHighly correlated pairs (|r| >= {high_threshold}) — consider dropping one:"
        )
        if high_pairs:
            for a, b, r in high_pairs:
                print(f"  {r:+.3f}  {a}  <->  {b}")
        else:
            print("  None")

        print(
            f"\nNear-zero correlation pairs (|r| <= {low_threshold}) — potentially uninformative:"
        )
        if low_pairs:
            for a, b, r in low_pairs[:10]:  # cap output
                print(f"  {r:+.3f}  {a}  <->  {b}")
            if len(low_pairs) > 10:
                print(f"  ... and {len(low_pairs) - 10} more")
        else:
            print("None")


if __name__ == "__main__":
    players_df = get_all_players(official_base_url)
    total_players = len(players_df)
    # asyncio.run(season_stats(total_players, players_df))
    df = asyncio.run(gameweek_stats(total_players, players_df))
    build_gameweek_level_df(df)

    # Exploratory checks on the gameweek-level features
    # exploratory_check()
