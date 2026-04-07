import time

from rich import console, print
from rich.live import Live
from rich.table import Table

from prediction.src.predict import (
    calc_accuracy,
    calc_brier,
    init_calc_accuracy,
    init_log_results,
    log_results,
    plot_calibration,
    predict,
)
from teams.src.team_stats import team_stats
from utils import get_gameweek, load_results

if __name__ == "__main__":
    con = console.Console()

    # print("Loading cached results...")
    # load_results()

    # con.log("Fetching team stats...")
    # team_stats()

    con.log("Calculating accuracy of predictions...")
    # # init_calc_accuracy()
    calc_accuracy()

    con.log("Logging prediction results...")
    # # init_log_results()
    log_results()

    con.log("Calculating Brier Score of the model.")
    calc_brier()

    con.log("Performing prediction")
    predict()

    con.log("Plotting calibration curves...")
    plot_calibration()
