# May need to fetch again whenever there are bgw or dgw
import json
import os
import sys

import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from const import TEAMS_DATA_DIR, leagues  # noqa: E402
from utils import read_fotmob_schedule  # noqa: E402


def get_pl_fixtures():
    official_api = "https://fantasy.premierleague.com/api/fixtures"
    flat_data = []
    with open("data/team_mapping.json", "r", encoding="utf-8") as f:
        team_map = json.load(f)

    try:
        print("Fetching fixtures from official api...")
        response = requests.get(official_api)
        data = response.json()

        for match in data:
            if match["event"]:

                home_team = team_map[str(match["team_h"])]
                away_team = team_map[str(match["team_a"])]
                flat_data.append(
                    {
                        "team": home_team,
                        "opponent": f"{away_team}-H",
                        "gw": match["event"],
                    }
                )
                # Away perspective
                flat_data.append(
                    {
                        "team": away_team,
                        "opponent": f"{home_team}-A",
                        "gw": match["event"],
                    }
                )

        df = pd.DataFrame(flat_data)

        fixture_df = df.pivot_table(
            index="team", columns="gw", values="opponent", aggfunc=list
        ).reset_index()

        file_path = TEAMS_DATA_DIR / "ENG_fixtures.csv"
        fixture_df.to_csv(file_path, index=False)

        print(f"Exported Premier League fixtures at {file_path}")
        print(fixture_df.tail(10))

    except requests.exceptions.RequestException as e:
        print(e)


def get_fixtures():
    print("Getting fixtures of premier league")
    get_pl_fixtures()

    for league in leagues:
        if league == "ENG-Premier League":
            continue
        flat_data = []
        league_code, league_name = league.split("-")
        print(f"Fetching {league_name} fixtures from FotMob...")
        schedule_df = read_fotmob_schedule(league)

        for _, row in schedule_df.iterrows():
            flat_data.append(
                {
                    "team": row["home_team"],
                    "opponent": f"{row['away_team']}-H",
                    "gw": row["round"],
                }
            )
            # Away perspective
            flat_data.append(
                {
                    "team": row["away_team"],
                    "opponent": f"{row['home_team']}-A",
                    "gw": row["round"],
                }
            )

        df = pd.DataFrame(flat_data)

        fixture_df = df.pivot_table(
            index="team", columns="gw", values="opponent", aggfunc=list
        ).reset_index()

        file_path = TEAMS_DATA_DIR / f"{league_code}_fixtures.csv"
        fixture_df.to_csv(file_path, index=False)

        print(f"Exported {league_name} fixtures at {file_path}")
        print(fixture_df.tail(10))


if __name__ == "__main__":
    get_fixtures()
