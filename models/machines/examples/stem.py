from models.machines.network import *
from models.machines.continuous import *
from models.machines.automata import *
from models.machines.utils import *

from matplotlib import rc

import matplotlib.pyplot as plt
import pandas
import numpy
import math
import seaborn

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 14})
rc('text', usetex=True)


def plot_stem_histogram(bandwidth, info, bw):
    bandwidth = bandwidth.rename({"time": "$time$"}, axis=1)
    ax = seaborn.kdeplot(data=bandwidth, x="bandwidth", hue="$time$")

    ax.set_ylabel("$prob(Q)$ ", fontsize=16)
    ax.set_xlabel("$Q$ $[1/seg]$", fontsize=16)
    ax.set_title("Stem State Bandwidth Distribution", fontsize=20)

    ax.set_xlim(0)

    plt.savefig('./models/papers/sysbio/figs/stem_cell_bandwidth' +
                "-I" + str(info) + "-P" + str(bw) + ".pdf", format='pdf', bbox_inches="tight")


def plot_stem_capsule(information, population, info, bw):
    population = population.set_index("time")

    ax = population.plot(figsize=(16, 8))

    ax.set_ylabel("$log_{2}(N_{s})$ ", fontsize=16)
    ax.set_xlabel("$t$ $[steps]$", fontsize=16)
    ax.set_title("Cell Population (log 2 scale)", fontsize=20)

    plt.savefig('./models/papers/sysbio/figs/stem_cell_pop' +
                "-I" + str(info) + "-P" + str(bw) + ".pdf", format='pdf', bbox_inches="tight")

    information = information.set_index("time")
    ax = information.plot(figsize=(16, 8))

    ax.set_ylabel("$Q(N_{s}) / Q$ ", fontsize=16)
    ax.set_xlabel("$t$ $[steps]$", fontsize=16)
    ax.set_title("Bandwidth Partition (log 2 scale)", fontsize=20)

    plt.savefig('./models/papers/sysbio/figs/stem_cell_capacity' + "-I" + str(info) + "-BW" + str(bw) + ".pdf",
                format='pdf', bbox_inches="tight")


def stem_capsule(info=0.5, bw=50):
    pop = 10
    stem = Machine(name="mother")

    stem.add_state("stem")
    stem.add_state("proliferate", flux=[Nu("proliferate"), Flow("grow")])
    stem.add_state("differentiate", flux=[Nu("differentiate"), Flow("grow")])
    stem.add_state("final", flux=Flow("chalone"))

    stem.add_flow("stem", "proliferate", flow=Flow("grow"))
    stem.add_flow("proliferate", "differentiate", flow=Flow("grow"))

    stem.add_flow("proliferate", "stem", flow=Flow("chalone"))

    stem.add_flow("differentiate", "final", flow=Flow("grow"))
    stem.add_flow("differentiate", "proliferate", flow=Flow("chalone"))

    bath = Bath(grid=1, machines=[stem], information=info, q=bw)

    position = numpy.array([0, 0])

    for i in range(0, pop):
        bath.instantiate(position, "mother", "proliferate")

    evolution = {"time": [], "stem": [], "proliferate": [], "differentiate": [], "final": []}
    capacity = {"time": [], "stem": [], "proliferate": [], "differentiate": [], "final": []}

    for cycle in range(0, 1):
        # run
        bath.run()

        # agent's q
        q, total = {s: 0 for s in bath.get_cell(position).state}, 0
        for agent in bath.get_cell(position).get_children():
            q[agent.state.symbol] += agent.q
            total += agent.q

        for each in q:
            if q[each] > 0:
                capacity[each].append(q[each] / total)
            else:
                capacity[each].append(0)

        state = bath.get_cell(position).state
        for each in state:
            if state[each] > 0:
                evolution[each].append(math.log(state[each]))
            else:
                evolution[each].append(0)

        evolution["time"].append(bath.get_time())
        capacity["time"].append(bath.get_time())

        print("[=]  ", bath.get_time(), bath.get_cell(position).state, bath.get_cell(position).leak(), q, total)

        # get children
        children = bath.get_cell(position).get_children()

        # break cycles
        current = len(children)
        if current == 0:
            break

        # agent's q
        total = 0
        for agent in children:
            total += agent.q

        delta = (bw * pop - total)
        source = delta // current

        while delta > 0:
            for agent in children:
                take = min(source, delta)
                agent.q += take
                delta -= take

        assert delta == 0

        print("[+] cycle", cycle)
        stems_capacity = {"time": list(), "bandwidth": list()}
        for step in range(0, 500):
            # run
            bath.run()

            # agent's q
            q, total = {s: 0 for s in bath.get_cell(position).state}, 0
            for agent in bath.get_cell(position).get_children():
                q[agent.state.symbol] += agent.q
                total += agent.q

            for each in q:
                if q[each] > 0:
                    capacity[each].append(q[each] / total)
                else:
                    capacity[each].append(0)

            state = bath.get_cell(position).state
            for each in state:
                if state[each] > 0:
                    evolution[each].append(math.log(state[each]))
                else:
                    evolution[each].append(0)

            evolution["time"].append(bath.get_time())
            capacity["time"].append(bath.get_time())

            print("[=]  ", bath.get_time(), bath.get_cell(position).state, bath.get_cell(position).leak(), q, total)

            if step % 50 == 0:
                # bandwidth distribution
                for agent in bath.get_cell(position).get_children():
                    if agent.state.symbol == "stem":
                        stems_capacity["bandwidth"].append(agent.q)
                        stems_capacity["time"].append(step)

        distribution = pandas.DataFrame(stems_capacity, columns=list(stems_capacity.keys()))
        plot_stem_histogram(distribution, info, pop * bw)

        print(distribution)

        # cut
        for agent in bath.get_cell(position).get_children():
            # random cut
            if random.uniform(0, 1) < 0.80:
                agent.gone = True

            # set grow
            bath.get_cell(position).bus["grow"] = pop * bw

    population = pandas.DataFrame(evolution, columns=list(evolution.keys()))
    information = pandas.DataFrame(capacity, columns=list(capacity.keys()))

    plot_stem_capsule(information, population, info, pop * bw)
    print(population)
    print(information)


def stem_capsule_morphogen():
    stem = Machine(name="mother")

    stem.add_state("stem")
    stem.add_state("renew", flux=[Nu("on", value=5), Nu("stem")])
    stem.add_state("differentiate", flux=[Nu("cell")])
    stem.add_state("cell", flux=Flow("chalone", value=2))

    stem.add_flow("stem", "renew", flow=Flow("grow"))

    stem.add_flow("renew", "stem", flow=Flow("chalone"))
    stem.add_flow("renew", "differentiate", flow=Flow("grow"))

    stem.add_flow("differentiate", "cell", flow=Flow("chalone"))

    morphogen = Machine(name="morphogen")

    morphogen.add_state("on", flux=Flow("grow"))
    morphogen.add_state("off", flux=Flow("chalone"))

    morphogen.add_flow("on", "off", flow=Flow("chalone", value=2))
    morphogen.add_flow("off", "on", flow=Flow("grow"))

    bath = Bath(grid=1, machines=[stem, morphogen])
    position = numpy.array([0, 0])

    bath.instantiate(position, "mother", "stem")
    bath.instantiate(position, "mother", "stem")
    bath.instantiate(position, "mother", "stem")
    bath.instantiate(position, "morphogen", "on")

    Plotter.show(stem)
    Plotter.show(morphogen)


def example():
    particle = Machine(name="particle")

    particle.add_state("s0", flux=[Flow("z0"), Flow("z1")])
    particle.add_state("s1", flux=[Flow("z0")])
    particle.add_state("s2")

    particle.add_flow("s0", "s1", flow=Flow("z0"))
    particle.add_flow("s1", "s2", flow=Flow("z1"))
    particle.add_flow("s2", "s0", flow=Flow("z0"))

    bath = Bath(grid=1, machines=[particle], information=1, q=10)
    position = numpy.array([0, 0])

    bath.instantiate(position, "particle", "s0")
    bath.instantiate(position, "particle", "s1")
    bath.instantiate(position, "particle", "s1")
    bath.instantiate(position, "particle", "s1")
    bath.instantiate(position, "particle", "s1")

    transport = Punctual(initial={"s0": 3, "s1": 2}, rates={"r_z0": 10, "a_z1": 10}, machines=[particle])
    sol = transport.solve()

    print(sol)


def stem_transport():
    stem = Machine(name="mother")

    stem.add_state("stem")
    stem.add_state("proliferate", flux=[Nu("proliferate"), Flow("grow")])
    stem.add_state("differentiate", flux=[Nu("differentiate"), Flow("grow")])
    stem.add_state("final", flux=Flow("chalone"))

    stem.add_flow("stem", "proliferate", flow=Flow("grow"))
    stem.add_flow("proliferate", "differentiate", flow=Flow("grow"))

    stem.add_flow("proliferate", "stem", flow=Flow("chalone"))

    stem.add_flow("differentiate", "final", flow=Flow("grow"))
    stem.add_flow("differentiate", "proliferate", flow=Flow("chalone"))

    transport = Punctual(initial={"proliferate": 1},
                         rates={"nu_proliferate": 0.05, "nu_differentiate": 0.10, "r_grow": 1.20, "r_chalone": 0.55},
                         machines=[stem], time_span=numpy.linspace(0, 50, 100))
    sol = transport.solve()

    print(sol)

    sol[["stem", "proliferate", "differentiate", "final"]].plot()
    plt.show()


def main():
    stem_transport()


main()