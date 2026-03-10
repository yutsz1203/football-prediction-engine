import json

import pandas as pd
import soccerdata as sd

from const import TEAMS_DATA_DIR, TEAMS_RESULTS_DIR, leagues, season
from utils import get_gameweek

"""
gather latest team season + last 5 stats
update schedules and results
"""


def team_stats(n=5):
    gameweeks = get_gameweek()
    for league in leagues:
        league_code, league_name = league.split("-")
        gw = int(gameweeks[league])
        current_season_team = []
        lastngames_team = []

        ss = sd.Sofascore(leagues=league, seasons=season)
        hist = ss.read_schedule()
        cols = ["home_team", "away_team", "home_score", "away_score"]
        hist_df = hist[cols].copy()
        hist_df.dropna(inplace=True)
        hist_df.reset_index(drop=True, inplace=True)

        with open(TEAMS_DATA_DIR / "teams.json", "r", encoding="utf-8") as file:
            teams = json.load(file)

        for team in teams[league_name]:
            season_df = hist_df.loc[
                (hist_df["home_team"] == team) | (hist_df["away_team"] == team)
            ]
            season_home_df = season_df.loc[season_df["home_team"] == team]
            season_away_df = season_df.loc[season_df["away_team"] == team]

            h_games = len(season_home_df)
            h_gf = season_home_df["home_score"].sum()
            h_ga = season_home_df["away_score"].sum()

            a_games = len(season_away_df)
            a_gf = season_away_df["away_score"].sum()
            a_ga = season_away_df["home_score"].sum()

            current_season_team.append(
                {
                    "team": team,
                    "games": len(season_df),
                    "gf": h_gf + a_gf,
                    "ga": h_ga + a_ga,
                    "h_games": h_games,
                    "h_gf": h_gf,
                    "h_ga": h_ga,
                    "a_gf": a_gf,
                    "a_ga": a_ga,
                    "a_games": a_games,
                }
            )

            lastn_df = season_df.tail(min(n, gw))
            lastn_home_df = lastn_df.loc[lastn_df["home_team"] == team]
            lastn_away_df = lastn_df.loc[lastn_df["away_team"] == team]

            lastn_h_games = len(lastn_home_df)
            lastn_h_gf = lastn_home_df["home_score"].sum()
            lastn_h_ga = lastn_home_df["away_score"].sum()

            lastn_a_games = len(lastn_away_df)
            lastn_a_gf = lastn_away_df["away_score"].sum()
            lastn_a_ga = lastn_away_df["home_score"].sum()

            lastngames_team.append(
                {
                    "team": team,
                    "games": len(lastn_df),
                    "gf": lastn_h_gf + lastn_a_gf,
                    "ga": lastn_h_ga + lastn_a_ga,
                    "h_games": lastn_h_games,
                    "h_gf": lastn_h_gf,
                    "h_ga": lastn_h_ga,
                    "a_gf": lastn_a_gf,
                    "a_ga": lastn_a_ga,
                    "a_games": lastn_a_games,
                }
            )

        current_season_df = pd.DataFrame(current_season_team)
        current_season_df.sort_values(by=["gf"], ascending=False, inplace=True)
        current_season_df.set_index("team", inplace=True)
        file_path = TEAMS_RESULTS_DIR / f"{league_code}_teams_currentseason.csv"
        current_season_df.to_csv(file_path)
        print(f"Team stats of current season saved to {file_path}")

        lastngames_df = pd.DataFrame(lastngames_team)
        lastngames_df.sort_values(by=["gf"], ascending=False, inplace=True)
        lastngames_df.set_index("team", inplace=True)
        file_path = TEAMS_RESULTS_DIR / f"{league_code}_teams_last{n}games.csv"
        lastngames_df.to_csv(file_path)
        print(f"Team stats of last {n} games saved to {file_path}")
