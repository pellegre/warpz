import numpy
import pandas

import seaborn
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.signal import find_peaks

from models.synchro.units import *

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14})
rc('text', usetex=True)
rc('figure', max_open_warning=0)

OUTPUT_FOLDER = "./models/synchro/output/"


def plot_frame(frame, name, legend, ax=None):
    observable = pandas.DataFrame()
    observable[legend] = frame[name]

    observable["time"] = frame["time"]
    observable = observable.set_index("time")
    return observable.plot(figsize=(16, 8), ax=ax)


def read_angles(name, run_id):
    filename = OUTPUT_FOLDER + name + "/run-" + str(run_id) + "/player_theta.csv"
    frame = pandas.read_csv(filename).to_dict()
    player_angles = {player: frame[player][0] for player in frame}

    return player_angles


def get_player_belief(name, player, run_id):
    filename = OUTPUT_FOLDER + name + "/run-" + str(run_id) + "/belief_frame.csv"
    frame = pandas.read_csv(filename)
    frame = frame[frame["player"] == player]
    frame = frame.drop("player", axis=1)

    return frame


def get_actions(name, columns, run_id):
    filename = OUTPUT_FOLDER + name + "/run-" + str(run_id) + "/action_frame.csv"
    frame = pandas.read_csv(filename)
    frame = frame[columns]

    return frame


def get_paths(name, run_id):
    filename = OUTPUT_FOLDER + name + "/run-" + str(run_id) + "/pointer_frame.csv"
    frame = pandas.read_csv(filename)

    return frame


def get_action_correlation(one: pandas.DataFrame, other: pandas.DataFrame):
    rows = other.columns
    columns = one.columns

    matrix = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)
    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            one_data, other_data = one[col].values, other[row].values
            n = len(one_data)

            z = numpy.sum((one_data - numpy.average(one_data)) * (other_data - numpy.average(other_data)))
            x = (n - 1) * numpy.std(one_data) * numpy.std(other_data)
            matrix[i, j] = z / x

    matrix /= numpy.sum(matrix[:, :])
    return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                            columns=[i + 1 for i in range(0, len(columns))])


def correlation(run_id=413):
    print("[+] correlation plotter")
    player_angles = read_angles("bayes-movable", run_id)

    followers = [player for player in player_angles if "player" in player]
    runners = [player for player in player_angles if "runner" in player]

    followers_frame = get_actions("bayes-movable", followers, run_id)
    runners_frame = get_actions("bayes-movable", runners, run_id)

    runners_correlation = get_action_correlation(runners_frame, runners_frame)
    followers_correlation = get_action_correlation(followers_frame, followers_frame)

    rf_correlation = get_action_correlation(followers_frame, runners_frame)

    plot = seaborn.heatmap(runners_correlation, linewidths=.5)
    plot.set_title("evaders correlation")
    plot.get_figure().savefig(OUTPUT_FOLDER + "/plots/evaders_correlation.pdf")
    plt.show()

    plot = seaborn.heatmap(followers_correlation, linewidths=.5)
    plot.set_title("followers correlation")
    plot.get_figure().savefig(OUTPUT_FOLDER + "/plots/followers_correlation.pdf")
    plt.show()

    plot = seaborn.heatmap(rf_correlation, linewidths=.5)
    plot.set_title("followers / evaders correlation")
    plot.get_figure().savefig(OUTPUT_FOLDER + "/plots/cross_correlation.pdf")
    plt.show()


def get_success_rate(one: pandas.DataFrame, other: pandas.DataFrame):
    rows = [p for i, p in enumerate(other.columns) if "player" in p or "runner" in p]
    columns = [p for i, p in enumerate(one.columns) if "player" in p or "runner" in p]

    delta = one["success"].values
    matrix = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)

    for i, row in enumerate(rows):
        for j, col in enumerate(columns):
            one_data, other_data = one[col].values, other[row].values

            z = numpy.sum((one_data * other_data) * delta)
            x = numpy.sum(one_data * other_data)
            matrix[i, j] = z / x

    matrix /= numpy.sum(matrix[:, :])
    return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                            columns=[i + 1 for i in range(0, len(columns))])


def get_cooperation(ac_frame, sr_frame):
    ac, sr = numpy.triu(ac_frame.values, 1).flatten(), numpy.triu(sr_frame.values, 1).flatten()
    ac, sr = ac[ac != 0], sr[sr != 0]
    np = len(ac)
    return (1 / (np - 1)) * numpy.sum((ac - numpy.mean(ac)) * (sr - numpy.mean(sr)))


def cooperation(run_id=413):
    print("[+] correlation plotter")

    player_angles = read_angles("bayes-movable", run_id)

    followers = [player for player in player_angles if "player" in player]
    runners = [player for player in player_angles if "runner" in player]

    followers_actions = get_actions("bayes-movable", followers, run_id)
    runners_actions = get_actions("bayes-movable", runners, run_id)

    paths = get_paths("bayes-movable", run_id)

    delta = paths["distance"].rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0])

    followers_actions["success"] = delta
    followers_actions = followers_actions.dropna()

    runners_actions["success"] = -delta
    runners_actions = runners_actions.dropna()

    player_angles = read_angles("bayes-movable", run_id)

    followers = [player for player in player_angles if "player" in player]
    runners = [player for player in player_angles if "runner" in player]

    followers_frame = get_actions("bayes-movable", followers, run_id)
    runners_frame = get_actions("bayes-movable", runners, run_id)

    runners_correlation = get_action_correlation(runners_frame, runners_frame)
    runners_success_rate = get_success_rate(runners_actions, runners_actions)

    followers_correlation = get_action_correlation(followers_frame, followers_frame)
    followers_success_rate = get_success_rate(followers_actions, followers_actions)

    print(get_cooperation(followers_correlation, followers_success_rate))
    print(get_cooperation(runners_correlation, runners_success_rate))

    rf_success_rate = get_success_rate(followers_actions, runners_actions)
    rf_correlation = get_action_correlation(followers_frame, runners_frame)
    print(get_cooperation(rf_correlation, rf_success_rate))


def get_synchronicity(one: pandas.DataFrame, other: pandas.DataFrame):
    rows = [p for i, p in enumerate(other.columns) if "player" in p or "runner" in p]
    columns = [p for i, p in enumerate(one.columns) if "player" in p or "runner" in p]

    synchro = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)

    for i, row in enumerate(rows):
        for m, col in enumerate(columns):
            one_data, other_data = one[col][one[col] != 0].index.values, other[row][other[row] != 0].index.values

            j = numpy.zeros(shape=(len(one_data) - 1, len(other_data) - 1), dtype=float)
            for k in range(0, len(one_data) - 1):
                for l in range(0, len(other_data) - 1):
                    tau = 0.5 * min([one_data[k + 1] - one_data[k], one_data[k] - one_data[k - 1],
                                     other_data[l + 1] - other_data[l], other_data[l] - other_data[l - 1]])
                    if 0 < one_data[k] - other_data[l] <= tau:
                        j[k, l] = 1
                    elif one_data[k] == other_data[l]:
                        j[k, l] = 1 / 2
                    else:
                        j[k, l] = 0

            synchro[i, m] = 2 * numpy.sum(j[k, l]) / math.sqrt(j.shape[0] * j.shape[1])

    return pandas.DataFrame(synchro, index=[i + 1 for i in range(0, len(columns))],
                            columns=[i + 1 for i in range(0, len(columns))])


def synchronicity(run_id=413):
    player_angles = read_angles("bayes-movable", run_id)

    followers = [player for player in player_angles if "player" in player]
    runners = [player for player in player_angles if "runner" in player]

    followers_actions = get_actions("bayes-movable", followers, run_id)
    runners_actions = get_actions("bayes-movable", runners, run_id)

    print(get_synchronicity(followers_actions.tail(100), followers_actions.tail(100)))

# cooperation()
# correlation()


synchronicity()
