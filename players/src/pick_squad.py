"""Quick GW1 squad picker.

Deliberately simple: predictor is last season's total FPL points, and the squad
is chosen by MILP under the real FPL constraints (budget, position quotas, max 3
per club, valid starting XI). No feature engineering, no positional models --
the GW-level models in model_results/ only beat a constant-prediction baseline by
~0.05 points per gameweek, so for an opening squad the extra machinery buys very
little. What actually matters is spending the 100.0 well, which is the MILP's job.

Players who did not play enough last season, and anyone without a name+pos match
in the 25/26 stats, are dropped -- that removes promoted clubs and new signings.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp
from scipy.sparse import csr_matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from const import PLAYERS_DATA_DIR, PLAYERS_RESULTS_DIR

BUDGET = 100.0
MIN_MINUTES = 1500
BENCH_WEIGHT = 0.1
SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11


def load_pool() -> pd.DataFrame:
    """New-season prices joined to last season's points, on name+pos."""
    new = pd.read_csv(PLAYERS_DATA_DIR / "players.csv")
    old = pd.read_csv(PLAYERS_RESULTS_DIR / "players_seasonstats_2526.csv")

    new = new.rename(columns={"Name": "name", "Pos": "pos", "Team": "team", "Cost": "cost"})
    new = new[["name", "pos", "team", "cost"]]
    old = old[["name", "pos", "points", "minutes", "starts"]]

    # 14 surnames repeat in the new file, so a name+pos key is not unique for
    # everyone. Drop the ambiguous ones rather than silently mismatching them.
    dup_new = new.duplicated(["name", "pos"], keep=False)
    dup_old = old.duplicated(["name", "pos"], keep=False)
    ambiguous = sorted(
        set(new.loc[dup_new, "name"]) | set(old.loc[dup_old, "name"])
    )
    new, old = new[~dup_new], old[~dup_old]

    pool = new.merge(old, on=["name", "pos"], how="inner")
    unmatched = len(new) - len(pool)
    pool = pool[pool["minutes"] >= MIN_MINUTES].reset_index(drop=True)

    print(f"pool: {len(pool)} players")
    print(f"  dropped {unmatched} with no 25/26 name+pos match (promoted clubs, new signings)")
    print(f"  dropped rest under {MIN_MINUTES} minutes")
    if ambiguous:
        print(f"  ambiguous duplicate names, excluded -- check by hand: {', '.join(ambiguous)}")
    return pool


def solve(pool: pd.DataFrame, force: list[str] | None = None) -> pd.DataFrame:
    """MILP over squad vars x (15 picked) and XI vars y (11 of them start).

    Names in `force` are pinned into the squad, overriding the value calculation.
    """
    force = force or []
    n = len(pool)
    pos = pool["pos"].to_numpy()
    points = pool["points"].to_numpy(float)

    # Bench players still occupy budget but rarely score, so discount them.
    obj = -np.concatenate([BENCH_WEIGHT * points, (1 - BENCH_WEIGHT) * points])

    rows, lo, hi = [], [], []

    def add(row: np.ndarray, lower: float, upper: float) -> None:
        rows.append(row)
        lo.append(lower)
        hi.append(upper)

    x_only = lambda v: np.concatenate([v, np.zeros(n)])  # noqa: E731
    y_only = lambda v: np.concatenate([np.zeros(n), v])  # noqa: E731

    add(x_only(np.ones(n)), 15, 15)
    add(x_only(pool["cost"].to_numpy(float)), 0, BUDGET)
    add(y_only(np.ones(n)), XI_SIZE, XI_SIZE)

    for p, quota in SQUAD_QUOTA.items():
        mask = (pos == p).astype(float)
        add(x_only(mask), quota, quota)
        add(y_only(mask), XI_MIN[p], XI_MAX[p])

    for club in pool["team"].unique():
        add(x_only((pool["team"] == club).to_numpy(float)), 0, MAX_PER_CLUB)

    for name in force:
        mask = (pool["name"] == name).to_numpy(float)
        if not mask.any():
            raise ValueError(f"{name!r} not in pool -- check spelling, minutes floor, ambiguous names")
        add(x_only(mask), 1, 1)

    # A player can only start if he is in the squad: y_i - x_i <= 0.
    for i in range(n):
        row = np.zeros(2 * n)
        row[i], row[n + i] = -1.0, 1.0
        add(row, -np.inf, 0)

    res = milp(
        c=obj,
        constraints=LinearConstraint(csr_matrix(np.array(rows)), lo, hi),
        integrality=np.ones(2 * n),
        bounds=(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"no feasible squad: {res.message}")

    picked = np.round(res.x[:n]).astype(bool)
    starting = np.round(res.x[n:]).astype(bool)

    squad = pool[picked].copy()
    squad["starting"] = starting[picked]
    order = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
    return squad.sort_values(
        ["starting", "pos", "points"],
        key=lambda c: c.map(order) if c.name == "pos" else c,
        ascending=[False, True, False],
    )


def report(squad: pd.DataFrame) -> None:
    xi, bench = squad[squad["starting"]], squad[~squad["starting"]]
    formation = "-".join(str((xi["pos"] == p).sum()) for p in ["DEF", "MID", "FWD"])

    print(f"\nStarting XI  ({formation})")
    for _, r in xi.iterrows():
        print(f"  {r['pos']:3s}  {r['name']:20s} {r['team']:15s} {r['cost']:5.1f}  {r['points']:4.0f}")
    print("Bench")
    for _, r in bench.iterrows():
        print(f"  {r['pos']:3s}  {r['name']:20s} {r['team']:15s} {r['cost']:5.1f}  {r['points']:4.0f}")

    captain = xi.loc[xi["points"].idxmax()]
    print(f"\ncost {squad['cost'].sum():.1f} / {BUDGET}   XI last-season points {xi['points'].sum():.0f}")
    print(f"captain: {captain['name']}")


if __name__ == "__main__":
    # Any names passed on the command line are pinned into the squad.
    report(solve(load_pool(), force=sys.argv[1:]))
