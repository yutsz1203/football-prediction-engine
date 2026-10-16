import ast
import json
from datetime import date
from functools import cache

import numpy as np
import pandas as pd
import requests
import soccerdata as sd
from scipy.stats import sem, t

from const import (
    TEAMS_DATA_DIR,
    fotmob_league_ids,
    leagues,
    official_base_url,
    season,
    team_name_map,
)

"""
Documentation: https://fbrapi.com/documentation

Endpoints:
/countries: GET method to retrieve football-related meta-data for all available countries
/leagues: GET method to retrieve meta-data for all unique leagues associated with a specified country.
/league-seasons: GET method to retrieve all season ids for a specific league id
/league-season-details: GET method to retrieve meta-data for a specific league id and season id.
/league-standings: GET method to retrieve all standings tables for a given league and season id.
/teams: GET method to retrieve team roster and schedule data
/players: GET method to retrieve player meta-data
/matches: GET method to retrieve to retrieve match meta-data.
/team-season-stats: GET method to retrieve season-level team statistical data for a specified league and season.
/team-match-stats: GET method to retrieve match-level team statistical data for a specified team, league and season.
/player-season-stats: GET method to retrieve aggregate season stats for all players for a team-league-season
/player-match-stats: GET method to retrieve matchlog data for a given player-league-season
/all-players-match-stats: GET method to retrieve match stats for all players in a match
"""

season_id = "2025-2026"
league_id = 9

# api = requests.post(base_url + generate_api_key)
# api_key = api.json()['api_key']
# print("API Key:", api_key)

# header = {"X-API-Key": api_key}
# from pathlib import Path

# print(Path(__file__).parent)
# df = pd.read_csv("players/results/player_season_stats.csv")
# print(df.columns)


def get_team_id_mapping():
    response = requests.get(f"{official_base_url}/bootstrap-static")

    data = response.json()
    teams = data["teams"]
    team_map = {}

    for team in teams:
        team_id = int(team["id"])
        # Adjust team name for MatchHistory api
        if team["name"] == "Man Utd":
            team_map[team_id] = "Man United"
        elif team["name"] == "Spurs":
            team_map[team_id] = "Tottenham"
        else:
            team_map[team_id] = team["name"]

    with open("data/team_mapping.json", "w", encoding="utf-8") as f:
        json.dump(team_map, f, indent=4)

    print("Output team mappings to data/team_mapping.json")


def get_teams():
    teams = {}
    for league in leagues:
        league_name = league.split("-")[1]
        mh = sd.MatchHistory(leagues=league, seasons=season)
        hist = mh.read_games()
        team_list = sorted(hist["home_team"].unique())
        teams[league_name] = team_list
        print(league)
        print(team_list)
        print("\n")

    with open("teams/data/teams.json", "w", encoding="utf-8") as file:
        json.dump(teams, file, indent=4)

    print("Output teams to teams/data/teams.json ")


def check_team_names(df: pd.DataFrame, source: str):
    """Raise if df has home/away team names that are not in teams/data/teams.json."""
    with open(TEAMS_DATA_DIR / "teams.json", "r", encoding="utf-8") as file:
        teams = json.load(file)
    unknown = set()
    for league, league_df in df.groupby("league"):
        league_teams = set(league_df["home_team"]) | set(league_df["away_team"])
        unknown |= league_teams - set(teams[league.split("-")[1]])
    if unknown:
        raise ValueError(
            f"Unknown {source} team names {sorted(unknown)}. Add them to team_name_map in const.py."
        )


def read_fotmob_schedule(league: str) -> pd.DataFrame:
    """Season fixtures of a league with kick-off time (UTC) and round number from FotMob."""
    response = requests.get(
        "https://www.fotmob.com/api/data/leagues",
        params={"id": fotmob_league_ids[league], "season": f"{season}/{season + 1}"},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    matches = response.json()["fixtures"]["allMatches"]

    schedule_df = pd.DataFrame(
        {
            "league": league,
            "date": pd.to_datetime(
                [match["status"]["utcTime"] for match in matches],
                utc=True,
                format="ISO8601",
            ),
            "home_team": [match["home"]["name"] for match in matches],
            "away_team": [match["away"]["name"] for match in matches],
            "round": [int(match["round"]) for match in matches],
        }
    )
    schedule_df[["home_team", "away_team"]] = schedule_df[
        ["home_team", "away_team"]
    ].replace(team_name_map)
    check_team_names(schedule_df, "FotMob")
    return schedule_df


def read_weeks() -> dict[tuple[str, str, str], int]:
    """Map (league, home_team, away_team) to its gameweek using teams/data/{code}_fixtures.csv."""
    weeks = {}
    for league in leagues:
        league_code = league.split("-")[0]
        fixtures_df = pd.read_csv(
            TEAMS_DATA_DIR / f"{league_code}_fixtures.csv", index_col="team"
        )
        for team, row in fixtures_df.iterrows():
            for gw, fixtures in row.dropna().items():
                for fixture in ast.literal_eval(fixtures):
                    opponent, side = fixture.rsplit("-", maxsplit=1)
                    if side == "H":
                        weeks[(league, team, opponent)] = int(gw)
    return weeks


@cache
def _load_schedule() -> pd.DataFrame:
    """
    Season schedule in the same shape as sd.Sofascore.read_schedule():
    fixtures and dates from FotMob, scores from football-data.co.uk (MatchHistory),
    gameweeks from the fixtures csv files.
    """
    schedule_df = pd.concat(
        [read_fotmob_schedule(league) for league in leagues], ignore_index=True
    )
    schedule_df["season"] = f"{season % 100:02d}{(season + 1) % 100:02d}"

    mh = sd.MatchHistory(leagues=leagues, seasons=season).read_games()
    results_df = (
        mh.reset_index()[["league", "home_team", "away_team", "FTHG", "FTAG"]]
        .rename(columns={"FTHG": "home_score", "FTAG": "away_score"})
        .dropna()
    )
    results_df[["home_team", "away_team"]] = results_df[
        ["home_team", "away_team"]
    ].replace(team_name_map)
    check_team_names(results_df, "football-data.co.uk")

    keys = ["league", "home_team", "away_team"]
    schedule_df = schedule_df.merge(results_df, on=keys, how="left", indicator=True)
    matched = (schedule_df["_merge"] == "both").sum()
    if matched != len(results_df):
        raise ValueError(
            f"Only {matched} of {len(results_df)} football-data.co.uk results matched a FotMob fixture."
        )

    weeks = read_weeks()
    schedule_df["week"] = [weeks.get(key) for key in schedule_df[keys].itertuples(index=False)]
    missing = schedule_df.loc[schedule_df["week"].isna(), keys]
    if not missing.empty:
        raise ValueError(
            f"No gameweek in the fixtures csv for:\n{missing.to_string(index=False)}"
        )
    schedule_df["week"] = schedule_df["week"].astype(int)

    schedule_df["game"] = (
        schedule_df["date"].dt.strftime("%Y-%m-%d")
        + " "
        + schedule_df["home_team"]
        + "-"
        + schedule_df["away_team"]
    )
    schedule_df.set_index(["league", "season", "game"], inplace=True)
    schedule_df.sort_index(inplace=True)
    return schedule_df[
        ["week", "date", "home_team", "away_team", "home_score", "away_score"]
    ]


def read_schedule(league: str | None = None) -> pd.DataFrame:
    """Unplayed matches have NaN scores. Pass a league to keep only its matches."""
    schedule = _load_schedule()
    if league:
        schedule = schedule.loc[schedule.index.get_level_values("league") == league]
    return schedule.copy()


def get_gameweek(type=None) -> dict:
    schedule = read_schedule()
    today = date.today().isoformat()
    gameweeks = {}

    for league in leagues:
        league_schedule = schedule.loc[league]
        league_schedule = league_schedule[~league_schedule.index.duplicated()]
        try:
            first_index = league_schedule.loc[
                league_schedule.index.get_level_values("game") >= today
            ]["week"].index.to_list()[0]
            first_index_loc = league_schedule.index.get_loc(first_index)
        except IndexError:
            first_index_loc = len(league_schedule) - 1

        if type == "prev":
            gw = str(league_schedule.iloc[first_index_loc - 1]["week"])
        else:
            if first_index_loc == len(league_schedule) - 1:
                gw = str(league_schedule.iloc[first_index_loc]["week"])
            else:
                gw = str(
                    max(
                        league_schedule.iloc[first_index_loc]["week"],
                        league_schedule.iloc[first_index_loc + 1]["week"],
                    )
                )
        gameweeks[league] = gw
    return gameweeks


def ci(data, confidence_level: float = 0.8) -> tuple[float, float]:
    degrees_of_freedom = len(data) - 1
    sample_mean = np.mean(data)
    sample_standard_error = sem(data)

    ci_lower, ci_upper = t.interval(
        confidence_level,
        degrees_of_freedom,
        loc=sample_mean,
        scale=sample_standard_error,
    )

    return round(ci_lower, 2), round(ci_upper, 2)
