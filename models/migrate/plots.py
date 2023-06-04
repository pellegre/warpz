from models.migrate.run import *

import seaborn
import matplotlib.pyplot as plt
import jenkspy

from matplotlib import rc
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier

from sklearn import linear_model
from scipy import stats

from scipy import stats

from matplotlib import pyplot

from sklearn.linear_model import LinearRegression

from matplotlib import style
style.use('seaborn-deep')

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14})
rc('text', usetex=True)
rc('figure', max_open_warning=0)

OUTPUT_FOLDER = "./models/migrate/output/"
RUNS = math.inf

MIN_CUTS = 3
MAX_CUT = 14


def process(folder="cut-06-12"):
    print("[+] process migrate simulation")

    # simulations
    simulations = dict()

    counter = 0
    for each_run in os.listdir(OUTPUT_FOLDER + folder):
        # get run ID
        run_id = each_run.split('-')[1]

        try:
            # get tally frame
            tally_frame = RunInstance.get_tally_frame(folder, run_id, output=OUTPUT_FOLDER)

            # observables frame
            observables_frame = tally_frame.genes_frame

            # get maximum time
            maximum_time = observables_frame["time"].max()

            if maximum_time > 2000 and maximum_time is not numpy.nan:
                print("[+] max time for", run_id, "=", maximum_time)
                simulations[run_id] = tally_frame
                counter += 1

        except FileNotFoundError:
            print("[+] file not found", run_id)

        if counter > RUNS:
            break

    # get back simulations results
    return simulations


def plot_frame(frame, name, legend, log=False, ax=None, x_axis="time", **kwargs):
    observable = pandas.DataFrame()
    if log:
        observable[legend] = numpy.log(frame[name].values)
    else:
        observable[legend] = frame[name]

    observable[x_axis] = frame[x_axis]
    observable = observable.set_index(x_axis)
    return observable.plot(figsize=(16, 8), ax=ax, **kwargs)


WARM_UP_PERIOD = 0
MAX_COVERAGE = 0.99
DELTA_COVERAGE = 0.05


def get_growth_time(surgery_time, coverage):
    t = surgery_time

    # go until tissue reaches full coverage
    while t in coverage.index and coverage.loc[t].values[0] < MAX_COVERAGE:
        t += 1

    return t - surgery_time


def get_growth_time_frame(observables):
    # infer surgery time and calculate re-growth time
    coverage = observables[observables["time"] > WARM_UP_PERIOD][["time", "coverage"]]
    coverage = coverage.set_index("time")
    surgery_time = coverage[coverage["coverage"].diff() < -DELTA_COVERAGE].index.values

    # prepare data
    growth_time = [[i + 1, get_growth_time(point, coverage)] for i, point in enumerate(surgery_time)][:-1]

    # get growth time frame
    growth_frame = pandas.DataFrame(growth_time, columns=["batch", "growth-time"])

    # return frame
    return growth_frame


def plot_growth_time(observables, legend=None, ax=None, log=False, **kwargs):
    # get growth frame
    growth_frame = get_growth_time_frame(observables)

    if len(growth_frame):
        # plot it
        ax = plot_frame(growth_frame, name="growth-time", style='.', linestyle='--',
                        x_axis="batch", ax=ax, legend=legend, log=log, **kwargs)
        ax.set_ylabel("Growth Time ($t$)")
        ax.set_xlabel("Cut Number ($n$)")
        ax.set_title("Growth time response after successive cuts (\%80 of the tissue)")

    plt.grid(True, color='gray', linestyle='--')

    return ax


def get_growth_time_response(tissue_simulations):
    plot_growth_time(tissue_simulations["370"].observables_frame, legend="Simulation \#370", ms=20)
    plt.savefig("./models/migrate/docs/figs/growth_time_370.pdf", format="pdf", bbox_inches="tight")


def get_average_growth_time(tissue_simulations):
    # get GF data frame
    gf_matrix = pandas.DataFrame()
    for each in tissue_simulations:
        gf = get_growth_time_frame(tissue_simulations[each].observables_frame)
        if len(gf) > MIN_CUTS:
            # get growth frame
            print("[+] got growth frame for", each, len(gf))
            gf.columns = ["batch", "run-" + each]
            gf = gf.set_index("batch")

            # concatenate to the matrix
            gf_matrix = pandas.concat([gf_matrix, gf], axis=1)

    print("[+] got GF matrix")
    gf_matrix = gf_matrix.dropna()
    print(gf_matrix)

    # get average time per cut
    legend = "$\hat{x}_{growth\;time}(n)$"
    cut_frame = pandas.DataFrame(columns=["cut", legend, "std"])
    for cut in gf_matrix.index:
        # get growth time values
        row = gf_matrix.loc[cut]
        values = [x for x in row.values if not numpy.isnan(x)]

        # get average (and deviation)
        print("[+] average of cut", cut, numpy.average(values), numpy.std(values))
        cut_frame = pandas.concat([cut_frame,
                                   pandas.DataFrame([[cut, numpy.average(values), numpy.std(values)]],
                                                    columns=cut_frame.columns)])

    cut_frame = cut_frame.set_index("cut")

    print("[+] got cut frame")
    print(cut_frame)

    ax = cut_frame.plot(kind="bar", y=legend,
                        title="Mean growth time response after successive cuts")
    ax.set_xlabel("Cut Number ($n$)")
    ax.set_ylabel("Growth Time ($t$)")

    for key, spine in ax.spines.items():
        if key != "bottom":
            spine.set_visible(False)

    ax.tick_params(left=False)
    ax.errorbar([i - 1 for i in cut_frame.index], cut_frame[legend], yerr=cut_frame["std"],
                linewidth=1.5, color="black", alpha=0.4, capsize=4)

    plt.xticks(rotation='horizontal')

    plt.show()


def linear_fit(x_train, y_train):
    reg = LinearRegression().fit(x_train, y_train)

    sse = numpy.sum((reg.predict(x_train) - y_train) ** 2, axis=0) / float(x_train.shape[0] - x_train.shape[1])
    print(sse)
    se = numpy.array([
        numpy.sqrt(numpy.diagonal(sse[i] * numpy.linalg.inv(numpy.dot(x_train.T, x_train))))
        for i in range(0, sse.shape[0])
    ])

    t = reg.coef_ / se
    p = 2 * (1 - stats.t.cdf(numpy.abs(t), y_train.shape[0] - x_train.shape[1]))

    return reg, p


def plot_growth_time_trend(tissue_simulations):
    # get GF data frame
    gf_matrix = pandas.DataFrame()
    for each in tissue_simulations:
        gf = get_growth_time_frame(tissue_simulations[each].observables_frame)
        if len(gf) > MIN_CUTS:
            # get growth frame
            print("[+] got growth frame for", each, len(gf))
            gf = gf.set_index("batch")
            gf["growth-time"] = numpy.log(gf["growth-time"].values)
            gf["run"] = each

            # concatenate to the matrix
            gf_matrix = pandas.concat([gf_matrix, gf], axis=0)

    faster_run = gf_matrix[gf_matrix["growth-time"] == gf_matrix["growth-time"].min()]["run"].values[0]
    print("[+] minimum growth time", faster_run)

    print("[+] got GF matrix")
    fast_tissues = gf_matrix[gf_matrix.index >= MAX_CUT]
    gf_matrix = gf_matrix[gf_matrix.index < MAX_CUT]

    # apply linear regression
    x_train, y_train = gf_matrix.index.values, gf_matrix["growth-time"].values
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_train, y_train)

    print("[+] linear regression")
    print("[+] slope", slope)
    print("[+] intercept", intercept)
    print("[+] p-value (regression)", p_value)
    print("[+] rvalue", r_value)

    gf_matrix["predicted"] = gf_matrix.index * slope + intercept
    gf_matrix["residues"] = gf_matrix["growth-time"] - gf_matrix["predicted"]

    print("[+] shapiro test")
    shapiro_test = stats.shapiro(gf_matrix["residues"].values)
    print("[+] p-value (shapiro)", shapiro_test[1])

    # plot residues
    ax = gf_matrix["residues"].plot(figsize=(16, 8), style=".", ms=12)
    ax.set_ylabel("Residue ($t$)")
    ax.set_xlabel("Predicted ($t$)")
    ax.set_title("Residues dispersion graph")

    plt.grid(True, color='gray', linestyle='--')

    plt.savefig("./models/migrate/docs/figs/residues_growth_time.pdf", format="pdf", bbox_inches="tight")

    plt.clf()

    # plot linear regression
    gf_matrix.index = gf_matrix.index.values.astype(int)
    ax = gf_matrix["growth-time"].plot(figsize=(16, 8), style=".", ms=12)
    ax = gf_matrix["predicted"].plot(style="-", ax=ax)
    ax = fast_tissues["growth-time"].plot(style=".", ms=8, ax=ax)
    ax.set_ylabel("Growth Time (log scale)")
    ax.set_title("Growth Time Response vs Cut $\\beta_{slope} = " +
                 f"{slope:.2f}\;pvalue (<0.0001)$")

    # plot it
    faster_growth = gf_matrix[gf_matrix["run"] == "370"]
    faster_growth["growth-time"].plot(style="--", ax=ax)

    ax.set_xlabel("Cut Number ($n$)")

    plt.grid(True, color='gray', linestyle='--')

    plt.savefig("./models/migrate/docs/figs/growth_time_regression.pdf", format="pdf", bbox_inches="tight")

    print("[+] got GF matrix")
    print(gf_matrix)


def get_importance(model, feature_names):
    importance = pandas.DataFrame(numpy.transpose(model.feature_importances_),
                                   columns=["importance"])
    importance.index = feature_names
    importance = importance.sort_values("importance", ascending=False)

    return importance


def plot_full_importance(tissue_simulations):
    # get GF data frame
    gf_matrix = pandas.DataFrame()
    final_growth = pandas.DataFrame(columns=["run", "time"], dtype=int)
    for each in tissue_simulations:
        gf = get_growth_time_frame(tissue_simulations[each].observables_frame)
        if len(gf) > MIN_CUTS:
            row = pandas.DataFrame([[each, gf["growth-time"].values[-1]]], columns=["run", "time"], dtype=int)
            final_growth = pandas.concat([final_growth, row])

            # get growth frame
            print("[+] got growth frame for", each, len(gf))
            gf.columns = ["batch", "run-" + each]
            gf = gf.set_index("batch")

            # concatenate to the matrix
            gf_matrix = pandas.concat([gf_matrix, gf], axis=1)

    print("[+] got GF matrix")
    print(gf_matrix)
    print(final_growth)

    breaks = jenkspy.jenks_breaks(final_growth["time"], nb_class=2)
    middle = breaks[1]

    print(breaks)

    class_array = numpy.ones(len(final_growth), dtype=int)
    class_array[final_growth["time"].values >= middle] = 0

    final_growth["class"] = class_array
    print("[+] final growth matrix")
    print(final_growth)

    genes_train = pandas.DataFrame()
    for each in tissue_simulations:
        genes_frame = tissue_simulations[each].genes_frame
        genes_frame = genes_frame[genes_frame["time"] > 12500]

        run_time = final_growth[final_growth["run"] == each]["class"]
        if len(run_time):
            class_value = run_time.values[0]
            genes_frame = genes_frame.assign(partition=numpy.array([class_value] * len(genes_frame)))

            genes_train = pandas.concat([genes_train, genes_frame])

    print("[+] got train frame")
    genes_train = genes_train.drop(["time"], axis=1)

    genes_data = genes_train[[c for c in genes_train.columns if c != "partition"]]
    genes_classes = genes_train["partition"]

    random_forest = RandomForestClassifier(n_estimators=200, n_jobs=8)
    random_forest.fit(genes_data.values, genes_classes.values)

    prediction = random_forest.predict(genes_data.values)
    print("[+] training accuracy", numpy.count_nonzero(prediction == genes_classes.values) / len(genes_classes.values))
    importance = get_importance(random_forest, [c.split(".")[1] for c in genes_data.columns])

    importance = importance[importance["importance"] > 0.01]
    ax = importance.plot(kind="barh", y="importance", title="Gene Importance (Random Forest)")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.get_legend().remove()

    plt.savefig("./models/migrate/docs/figs/importance.pdf", format="pdf", bbox_inches="tight")

    plt.clf()

    for key, spine in ax.spines.items():
        if key != "bottom":
            spine.set_visible(False)

    ax.tick_params(bottom=False)

    for e in range(0, len(importance.index)):
        max_importance = importance.index[e]
        print("[+] got importance")
        print(importance)
        print("[+] max importance", max_importance)

        fast_class = genes_train[genes_train["partition"] == 1]["Trait." + max_importance]
        slow_class = genes_train[genes_train["partition"] == 0]["Trait." + max_importance]

        ax = seaborn.distplot(fast_class, label='fast')
        ax = seaborn.distplot(slow_class, label='slow', ax=ax)
        ax.legend(loc='upper left')

        ax.set_xlabel("Gene Value (bit)")
        ax.set_ylabel("Density")
        ax.set_title(max_importance + " Gene Distribution")

        plt.savefig("./models/migrate/docs/figs/distribution_" + str(max_importance).lower() + ".pdf",
                    format="pdf", bbox_inches="tight")

        plt.clf()


def main():
    print("[+] plots")

    # read simulations
    tissue_simulations = process()
    print("[+] got", len(tissue_simulations), "simulations")

    # get GF data frame
    gf_matrix = pandas.DataFrame()
    final_growth = pandas.DataFrame(columns=["run", "time"], dtype=int)
    for each in tissue_simulations:
        gf = get_growth_time_frame(tissue_simulations[each].observables_frame)
        if len(gf) > MIN_CUTS:
            row = pandas.DataFrame([[each, gf["growth-time"].values[-1]]], columns=["run", "time"], dtype=int)
            final_growth = pandas.concat([final_growth, row])

            # get growth frame
            print("[+] got growth frame for", each, len(gf))
            gf.columns = ["batch", "run-" + each]
            gf = gf.set_index("batch")

            # concatenate to the matrix
            gf_matrix = pandas.concat([gf_matrix, gf], axis=1)

    print("[+] got GF matrix")
    print(gf_matrix)
    print(final_growth)

    breaks = jenkspy.jenks_breaks(final_growth["time"], nb_class=2)
    middle = breaks[1]

    print(breaks)

    class_array = numpy.ones(len(final_growth), dtype=int)
    class_array[final_growth["time"].values >= middle] = 0

    final_growth["class"] = class_array
    print("[+] final growth matrix")
    print(final_growth)

    genes_train = pandas.DataFrame()
    for each in tissue_simulations:
        genes_frame = tissue_simulations[each].genes_frame
        genes_frame = genes_frame[genes_frame["time"] > 12500]

        run_time = final_growth[final_growth["run"] == each]["class"]
        if len(run_time):
            class_value = run_time.values[0]
            genes_frame = genes_frame.assign(partition=numpy.array([class_value] * len(genes_frame)))

            genes_train = pandas.concat([genes_train, genes_frame])

    print("[+] got train frame")
    genes_train = genes_train.drop(["time"], axis=1)

    slow_class = genes_train[genes_train["partition"] == 0]

    ax = seaborn.distplot(slow_class["Trait.PRODUCTION"], hist=False, kde=True, label='production')
    ax = seaborn.distplot(slow_class["Trait.RELEASE"], hist=False, kde=True, label='release', ax=ax)
    ax = seaborn.distplot(slow_class["Trait.CONSUMPTION"], hist=False, kde=True, label='consumption', ax=ax)
    ax.legend(loc='upper right')

    ax.set_xlabel("Rate $units/time$")
    ax.set_ylabel("Density")
    ax.set_title("Bipolar chemokine rates (slow tissues)")

    plt.savefig("./models/migrate/docs/figs/rate_slow.pdf", format="pdf", bbox_inches="tight")

    plt.clf()

    fast_class = genes_train[genes_train["partition"] == 1]

    ax = seaborn.distplot(fast_class["Trait.PRODUCTION"], hist=False, kde=True, label='production')
    ax = seaborn.distplot(fast_class["Trait.RELEASE"], hist=False, kde=True, label='release', ax=ax)
    ax = seaborn.distplot(fast_class["Trait.CONSUMPTION"], hist=False, kde=True, label='consumption', ax=ax)
    ax.legend(loc='upper left')

    ax.set_xlabel("Rate $units/time$")
    ax.set_ylabel("Density")
    ax.set_title("Bipolar chemokine rates (fast tissues)")

    plt.savefig("./models/migrate/docs/figs/rate_fast.pdf", format="pdf", bbox_inches="tight")

    plt.clf()

    max_importance = "MUTATION"
    print("[+] got importance")
    print("[+] max importance", max_importance)

    fast_class = genes_train[genes_train["partition"] == 1]["Trait." + max_importance]
    slow_class = genes_train[genes_train["partition"] == 0]["Trait." + max_importance]

    ax = seaborn.distplot(fast_class, label='fast')
    ax = seaborn.distplot(slow_class, label='slow', ax=ax)
    ax.legend(loc='upper right')

    ax.set_xlabel("Gene Value (bit)")
    ax.set_ylabel("Density")
    ax.set_title(max_importance + " Gene Distribution")

    plt.savefig("./models/migrate/docs/figs/distribution_" + str(max_importance).lower() + ".pdf",
                format="pdf", bbox_inches="tight")

    plt.clf()


main()
