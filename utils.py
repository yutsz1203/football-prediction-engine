import json
from datetime import date

import numpy as np
import requests
import soccerdata as sd
from scipy.stats import sem, t

from const import leagues, official_base_url, season

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


def load_results():
    sofascore = sd.Sofascore(leagues=["ESP-La Liga", "GER-Bundesliga"], seasons=season)
    schedule = sofascore.read_schedule(force_cache=True)
    print(schedule.dropna())


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


def test():
    sofascore = sd.Sofascore(leagues="FRA-Ligue 1", seasons=season)
    schedule = sofascore.read_schedule(force_cache=True)

    gw = int(input(f"Enter gameweek: "))
    print(schedule.loc[schedule["week"] == gw])

    print(schedule.dtypes)


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


def get_gameweek(type=None) -> dict:
    sofascore = sd.Sofascore(leagues=leagues, seasons=season)
    schedule = sofascore.read_schedule(force_cache=True)
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
