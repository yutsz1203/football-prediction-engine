from prediction.src.predict import (
    calc_accuracy,
    init_calc_accuracy,
    init_log_results,
    log_results,
    plot_calibration,
    predict,
)
from teams.src.team_stats import team_stats
from utils import get_gameweek, load_results

if __name__ == "__main__":

    # print("Loading cached results...")
    # load_results()

    # print("Fetching team stats...")
    # team_stats()

    print("Calculating accuracy of predictions...")
    # # init_calc_accuracy()
    calc_accuracy()

    # print("Logging prediction results...")
    # # init_log_results()
    # log_results()

    # print("Plotting calibration curves...")
    # plot_calibration()

    # print("Performing prediction")
    # predict()
