# url: https://fantasy.premierleague.com/api/element-summary/{player-id}/
# or try this: https://fantasy.premierleague.com/api/event/1/live/
import asyncio
import os
import sys

import aiohttp
import pandas as pd
from rich.progress import Progress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from const import (  # noqa : E402
    PLAYERS_DATA_DIR,
    PLAYERS_RESULTS_DIR,
    official_element_summary,
)

MAX_CONCURRENT_REQUESTS = 10
MAX_RETRIES = 5
REQUEST_TIMEOUT_SECONDS = 30


def stats_calculation(matches):
    (
        points,
        games,
        goals,
        assists,
        xG,
        xA,
        xGI,
        gc,
        xGc,
        clean_sheets,
        defensive_contribution,
        saves,
        bonus,
    ) = (
        0,
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0,
        0,
        0,
        0,
    )
    (
        h_games,
        h_goals,
        h_assists,
        hxG,
        hxA,
        hxGI,
        h_gc,
        hxGc,
    ) = (
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
    )
    a_games, a_goals, a_assists, axG, axA, axGI, a_gc, axGc = (
        0,
        0,
        0,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
    )

    for match in matches:
        if match["minutes"] == 0:
            continue
        side = "Home" if match["was_home"] else "Away"
        points += match["total_points"]
        games += 1
        goals += match["goals_scored"]
        assists += match["assists"]
        xG += float(match["expected_goals"])
        xA += float(match["expected_assists"])
        xGI += float(match["expected_goal_involvements"])
        gc += match["goals_conceded"]
        xGc += float(match["expected_goals_conceded"])
        clean_sheets += match["clean_sheets"]
        defensive_contribution += match["defensive_contribution"]
        saves += match["saves"]
        bonus += match["bonus"]

        if side == "Home":
            h_games += 1
            h_goals += match["goals_scored"]
            h_assists += match["assists"]
            hxG += float(match["expected_goals"])
            hxA += float(match["expected_assists"])
            hxGI += float(match["expected_goal_involvements"])
            h_gc += match["goals_conceded"]
            hxGc += float(match["expected_goals_conceded"])
        else:
            a_games += 1
            a_goals += match["goals_scored"]
            a_assists += match["assists"]
            axG += float(match["expected_goals"])
            axA += float(match["expected_assists"])
            axGI += float(match["expected_goal_involvements"])
            a_gc += match["goals_conceded"]
            axGc += float(match["expected_goals_conceded"])

    return (
        points,
        games,
        goals,
        assists,
        xG,
        xA,
        xGI,
        gc,
        xGc,
        clean_sheets,
        defensive_contribution,
        saves,
        bonus,
        h_games,
        h_goals,
        h_assists,
        hxG,
        hxA,
        hxGI,
        h_gc,
        hxGc,
        a_games,
        a_goals,
        a_assists,
        axG,
        axA,
        axGI,
        a_gc,
        axGc,
    )


def build_dict(
    element_id,
    player,
    pos,
    team,
    cost,
    points,
    games,
    goals,
    assists,
    xG,
    xA,
    xGI,
    gc,
    xGc,
    clean_sheets,
    defensive_contribution,
    saves,
    bonus,
    h_games,
    h_goals,
    h_assists,
    hxG,
    hxA,
    hxGI,
    h_gc,
    hxGc,
    a_games,
    a_goals,
    a_assists,
    axG,
    axA,
    axGI,
    a_gc,
    axGc,
):
    return {
        "Player ID": element_id,
        "Name": player,
        "Pos": pos,
        "Team": team,
        "Cost": cost,
        "Total Points": points,
        "Bonus": bonus,
        "Points/$": round(points / cost, 2),
        "Games": games,
        "Goals": goals,
        "Assists": assists,
        "xG": round(xG, 2),
        "xA": round(xA, 2),
        "xGI": round(xGI, 2),
        "gc": gc,
        "xGc": round(xGc, 2),
        "clean_sheets": clean_sheets,
        "def_con": defensive_contribution,
        "Saves": saves,
        "h_Games": h_games,
        "h_Goals": h_goals,
        "h_Assists": h_assists,
        "h_xG": round(hxG, 2),
        "h_xA": round(hxA, 2),
        "h_xGI": round(hxGI, 2),
        "h_Gc": h_gc,
        "h_xGc": round(hxGc, 2),
        "a_Games": a_games,
        "a_Goals": a_goals,
        "a_Assists": a_assists,
        "a_xG": round(axG, 2),
        "a_xA": round(axA, 2),
        "a_xGI": round(axGI, 2),
        "a_Gc": a_gc,
        "a_xGc": round(axGc, 2),
    }


async def fetch_player_history(element_id, session, semaphore):
    """Fetch one player's match history.

    Returns the list of matches, or None if the player has no history (i.e. has
    not played in the current season). Raises RuntimeError once the retries are
    exhausted so the caller can report which players were lost.
    """
    url = f"{official_element_summary}/{element_id}/"
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                async with session.get(url) as response:
                    if response.status == 429:
                        last_error = "rate limited (HTTP 429)"
                        await asyncio.sleep(2**attempt)
                        continue
                    response.raise_for_status()
                    content = await response.json()
            return content.get("history")
        except Exception as e:
            last_error = repr(e)
            await asyncio.sleep(2**attempt)
    raise RuntimeError(
        f"gave up on player {element_id} after {MAX_RETRIES} attempts: {last_error}"
    )


async def fetch_all_histories(element_ids):
    """Fetch every player's history concurrently.

    Results come back in the same order as element_ids; each entry is either the
    match list, None (player has not played), or the Exception that ended it.
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        with Progress() as progress:
            task = progress.add_task("Fetching player data...", total=len(element_ids))

            async def fetch_and_advance(element_id):
                try:
                    return await fetch_player_history(element_id, session, semaphore)
                finally:
                    progress.update(task, advance=1)

            return await asyncio.gather(
                *(fetch_and_advance(element_id) for element_id in element_ids),
                return_exceptions=True,
            )


def build_row(player_basic_df, element_id, matches):
    """Turn one player's matches into a single output row."""
    (
        points,
        games,
        goals,
        assists,
        xG,
        xA,
        xGI,
        gc,
        xGc,
        clean_sheets,
        defensive_contribution,
        saves,
        bonus,
        h_games,
        h_goals,
        h_assists,
        hxG,
        hxA,
        hxGI,
        h_gc,
        hxGc,
        a_games,
        a_goals,
        a_assists,
        axG,
        axA,
        axGI,
        a_gc,
        axGc,
    ) = stats_calculation(matches)

    return build_dict(
        element_id,
        player_basic_df.loc[element_id, "Name"],
        player_basic_df.loc[element_id, "Pos"],
        player_basic_df.loc[element_id, "Team"],
        player_basic_df.loc[element_id, "Cost"],
        points,
        games,
        goals,
        assists,
        xG,
        xA,
        xGI,
        gc,
        xGc,
        clean_sheets,
        defensive_contribution,
        saves,
        bonus,
        h_games,
        h_goals,
        h_assists,
        hxG,
        hxA,
        hxGI,
        h_gc,
        hxGc,
        a_games,
        a_goals,
        a_assists,
        axG,
        axA,
        axGI,
        a_gc,
        axGc,
    )


def save(rows, file_path, description):
    df = pd.DataFrame(rows)
    df.sort_values(by=["xGI", "Points/$"], ascending=False, inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Player stats of {description} saved to {file_path}")
    return df


def main(n):
    player_basic_df = pd.read_csv(
        PLAYERS_DATA_DIR / "players.csv", index_col="Player ID"
    )
    player_ids = player_basic_df.index.tolist()

    histories = asyncio.run(fetch_all_histories(player_ids))

    df = []
    lastngamesdf = []
    not_played = []
    failed = []
    for element_id, history in zip(player_ids, histories):
        player = player_basic_df.loc[element_id, "Name"]
        if isinstance(history, BaseException):
            failed.append(f"{player} ({element_id}): {history}")
            continue
        if history is None:
            not_played.append(player)
            continue
        df.append(build_row(player_basic_df, element_id, history))
        lastngamesdf.append(build_row(player_basic_df, element_id, history[-n:]))

    if not_played:
        print(
            f"{len(not_played)} players have not played in the current season, "
            f"skipped: {', '.join(not_played)}"
        )
    if failed:
        print(f"\n{len(failed)} players could not be fetched and are MISSING:")
        for failure in failed:
            print(f"  - {failure}")

    df = save(
        df,
        PLAYERS_RESULTS_DIR / "players_currentseason.csv",
        "current season",
    )
    lastngamesdf = save(
        lastngamesdf,
        PLAYERS_RESULTS_DIR / f"players_last{n}games.csv",
        f"last {n} games",
    )

    print(df.head())
    print(lastngamesdf.head())


if __name__ == "__main__":
    n = int(input("Enter number of last n games to include in calculation: "))
    main(n)
