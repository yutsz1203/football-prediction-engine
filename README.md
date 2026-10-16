# Football Prediction Engine

Match predictions for Europe's top five leagues, plus Fantasy Premier League (FPL) tools built on top of them.

- **Match predictions**: home/draw/away, most likely score, and over/under 2.5 and 3.5 goals for every upcoming game in the Premier League, La Liga, Serie A, Bundesliga and Ligue 1.
- **FPL team forecasts**: expected goals for and against for each Premier League team over the next gameweek and the next 5 gameweeks.
- **Model tracking**: accuracy, Brier score and calibration of past predictions, updated as results come in.
- **FPL player tools**: player statistics, your own FPL team's statistics, and a GW1 squad picker.

## Table of contents

- [Football Prediction Engine](#football-prediction-engine)
  - [Table of contents](#table-of-contents)
  - [How the model works](#how-the-model-works)
  - [Data sources](#data-sources)
  - [Getting started](#getting-started)
  - [Weekly workflow](#weekly-workflow)
  - [Outputs](#outputs)
  - [FPL player tools](#fpl-player-tools)
  - [New season checklist](#new-season-checklist)
  - [Project layout](#project-layout)
  - [References](#references)

## How the model works

Each league gets its own Poisson regression (a GLM fitted with `statsmodels`), trained on the current season's results:

```
goals ~ home + team + opponent
```

- Every match gives two rows, one per side. `team` is the scoring side's attacking strength, `opponent` is the conceding side's defensive strength, and `home` is the home advantage.
- Matches are weighted by recency: weight = exp(−0.01 × days before the gameweek). A match played about 70 days earlier counts half as much as one played on the day.
- For each fixture, the model gives the expected goals for both sides. Two independent Poisson distributions (0–5 goals each) turn those into a score matrix, which gives the outcome probabilities, the most likely score and the over/under probabilities.

As results arrive, each prediction is scored against the actual result:

- **Accuracy**: share of correct outcome, over/under 2.5 and over/under 3.5 calls per league.
- **Brier score**: mean squared error of the predicted probabilities, per market.
- **Calibration**: predicted probabilities against observed frequencies, in 5 bins.

## Data sources

| Data | Source | Used for |
|---|---|---|
| Fixtures and kick-off times | [FotMob](https://www.fotmob.com) | Schedule, current gameweek, fixture lists for La Liga, Serie A, Bundesliga and Ligue 1 |
| Results | [football-data.co.uk](https://www.football-data.co.uk) (through `soccerdata.MatchHistory`) | Training data and scoring past predictions |
| Premier League gameweeks | [Official FPL API](https://fantasy.premierleague.com/api/) | FPL gameweek numbers, which include blank and double gameweeks |
| Players and FPL teams | Official FPL API | Player list, player statistics, your FPL team |

The gameweek of each match comes from `teams/data/{LEAGUE}_fixtures.csv`. For the Premier League, that file follows FPL gameweeks. For the other leagues, it follows FotMob's round numbers.


## Getting started

Requires Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Run every command from the project root. Some scripts read files with paths relative to it.

## Weekly workflow

Run these steps after each gameweek finishes.

**1. Update fixtures.** Postponed and rearranged games change the fixture lists, and in the Premier League they create blank and double gameweeks.

```bash
uv run teams/get_fixtures.py
```

This rewrites `teams/data/{ENG,ESP,FRA,GER,ITA}_fixtures.csv`.

**2. Run the pipeline.**

```bash
uv run main.py
```

This takes about 30 seconds and runs these steps in order:

1. **Team stats**: downloads the schedule and results, and writes goals for and against per team for the season and the last 5 games.
2. **Accuracy**: scores the saved predictions against new results. If a league has no saved predictions for its new results, for example at the start of a season, it is skipped.
3. **Log results**: adds each newly scored prediction to the log.
4. **Brier score**: recalculates the Brier score per market from the log.
5. **Predict**: fits the model for each league and predicts the coming gameweek.
6. **Calibration**: saves the calibration chart and opens it in a window. **The script finishes only when you close that window.**

## Outputs

| File | Contents |
|---|---|
| `teams/projection/nextforecast.csv` | Premier League: expected goals for and against per team for the coming gameweek |
| `teams/projection/next5forecast.csv` | Premier League: opponents and total expected goals for and against over the next 5 gameweeks |
| `prediction/data/{LEAGUE}/{LEAGUE}_gw{N}_long.json` | Full prediction per match: outcome probabilities, most likely score and its probability, over/under 2.5 and 3.5 |
| `prediction/data/{LEAGUE}/{LEAGUE}_gw{N}_short.json` | Short version: predicted outcome and over/under calls |
| `teams/results/{LEAGUE}_teams_currentseason.csv` | Games, goals for and goals against per team (overall, home and away) for the season |
| `teams/results/{LEAGUE}_teams_last5games.csv` | The same for each team's last 5 games |
| `prediction/output/accuracy.csv` | Cumulative accuracy per league |
| `prediction/output/log.csv` | Every scored prediction: probability, outcome (0/1) and market |
| `prediction/output/brier.json` | Brier score per market (`moneyline`, `2_5`, `3_5`) |
| `prediction/output/calibration_curves.png` | Calibration chart |

## FPL player tools

These scripts are separate from `main.py`. Run them when you need them.

**Update the player list.** New signings and academy promotions appear during the season.

```bash
uv run players/generate_df/get_players.py
```

Writes `players/data/players.csv`.

**Player statistics.** Asks how many recent games to include (5 is recommended).

```bash
uv run players/generate_df/player_stats.py
```

Writes `players/results/players_currentseason.csv` and `players/results/players_last{n}games.csv`, sorted by xGI and points per £m.

**Your FPL team's statistics.** Uses the FPL ID in `const.py` (`fpl_id`). Asks for the current gameweek and the same number of games as above.

```bash
uv run myteam/myteam_stats.py
```

Writes `myteam/results/myteam_currentseason_stats.csv` and `myteam/results/myteam_last{n}games_stats.csv`.

**GW1 squad picker.** Chooses a 15-player squad with mixed-integer linear programming. It maximises last season's FPL points under the real rules: £100m budget, position quotas, at most 3 players per club and a valid starting XI. Pass player names to force them into the squad.

```bash
uv run players/src/pick_squad.py
```

**Player points models (exploratory).** `players/src/player_stats.py` builds gameweek-level datasets per position in `players/results/gameweek_level/`. `players/src/model.py` fits OLS models on them, selects features by VIF, and evaluates them with an expanding window. Results go to `players/results/model_results/`.

## New season checklist

Before the first gameweek of a new season:

1. **`const.py`**: update `season` and `season_id`, `official_team_id_map`, and `fpl_id` if your FPL team changed.
2. **Team list**: run `get_teams()` in `utils.py` to regenerate `teams/data/teams.json`, and `get_team_id_mapping()` to regenerate `data/team_mapping.json`.
3. **Team names**: promoted teams often have different names in FotMob, football-data.co.uk and `teams.json`. Add each one to `team_name_map` in `const.py`. If a name is missing, the pipeline stops with an error that lists the unknown names.
4. **Fixtures**: run `uv run teams/get_fixtures.py`.
5. **Accuracy and log files**: `prediction/output/accuracy.csv` and `log.csv` carry over from last season. `init_calc_accuracy()` and `init_log_results()` in `prediction/src/predict.py` rebuild them; their lines in `main.py` are commented out.

## Project layout

```
main.py                     Weekly pipeline
const.py                    Paths, season settings, team name map, FPL API settings
utils.py                    Schedule loader (FotMob + football-data.co.uk), gameweek lookup, helpers
prediction/src/predict.py   Model, predictions, accuracy, log, Brier score, calibration
prediction/data/            Saved predictions per league and gameweek
prediction/output/          Accuracy, log, Brier score, calibration chart
teams/get_fixtures.py       Fixture lists per league
teams/src/team_stats.py     Goals for and against per team
teams/data/                 Fixture lists and teams.json
teams/results/              Team stats
teams/projection/           Premier League team forecasts
teams/preseason_prep/       Preseason team strength scripts (FBref API); run order in notes.md
players/                    FPL player list, statistics, squad picker, points models
myteam/                     Your FPL team's statistics
data/                       FPL team ID mapping and past-season team data
docs/                       Design notes
```

## References

- FotMob: https://www.fotmob.com
- football-data.co.uk: https://www.football-data.co.uk
- soccerdata: https://soccerdata.readthedocs.io
- Official FPL API: https://fantasy.premierleague.com/api/
- FPL API guide: https://www.oliverlooney.com/blogs/FPL-APIs-Explained#how-to-use-authenticated-endpoints
- FBref API (preseason scripts): https://fbrapi.com/documentation
