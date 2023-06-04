from models.machines.automata import *

import numpy


def capsule_testing_cycle():
    a = Machine(name="A")

    a.add_state("a_1", flux=[Flow("sx"), Flow("sz"), Flow("sz")])
    a.add_state("a_2", flux=[Flow("sz")])

    a.add_flow("a_1", "a_2", flow=Flow("sz"))
    a.add_flow("a_2", "a_1", flow=Flow("sx"))

    b = Machine(name="B")

    b.add_state("b_1", flux=[Flow("sz"), Flow("sx")])
    b.add_state("b_2", flux=[Flow("sx")])

    b.add_flow("b_1", "b_2", flow=Flow("sx"))
    b.add_flow("b_2", "b_1", flow=Flow("sz"))

    bath = Bath(grid=1, machines=[a, b])
    capsule = bath.get_cell([0, 0])

    bath.instantiate(numpy.array([0, 0]), "A", "a_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")

    # setup bus
    capsule.post()

    # check bus
    assert capsule.bus["sx"] == 2
    assert capsule.bus["sz"] == 3

    # check agents
    assert len(capsule.listening["sx"]) == 1
    assert next(iter(capsule.listening["sx"])).state.symbol == "b_1"
    assert len(capsule.listening["sz"]) == 1
    assert next(iter(capsule.listening["sz"])).state.symbol == "a_1"

    # setup bus
    capsule.read()

    # check bus
    assert capsule.dissipated == 3

    # check agents
    assert capsule.agents[0].state.symbol == "a_2"
    assert capsule.agents[1].state.symbol == "b_2"

    # second round
    capsule.post()
    capsule.read()

    # check bus
    assert capsule.dissipated == 3

    # check agents
    assert capsule.agents[0].state.symbol == "a_1"
    assert capsule.agents[1].state.symbol == "b_1"


def capsule_testing_distribution_1():
    a = Machine(name="A")

    a.add_state("a_1", flux=[Flow("sx"), Flow("sx"), Flow("sx"), Flow("sx"), Flow("sx")])
    a.add_state("a_2", flux=[Flow("sz")])

    a.add_flow("a_1", "a_2", flow=Flow("sz"))
    a.add_flow("a_2", "a_1", flow=Flow("sx"))

    b = Machine(name="B")

    b.add_state("b_1", flux=[Flow("sz")])
    b.add_state("b_2", flux=[Flow("sx")])

    b.add_flow("b_1", "b_2", flow=Flow("sx"))
    b.add_flow("b_2", "b_1", flow=Flow("sz"))

    c = Machine(name="C")

    c.add_state("c_1")
    c.add_state("c_2")

    c.add_flow("c_1", "c_2", flow=Flow("sx"))
    c.add_flow("c_2", "c_1", flow=Flow("sz"))

    bath = Bath(grid=1, machines=[a, b, c])
    capsule = bath.get_cell([0, 0])

    bath.instantiate(numpy.array([0, 0]), "A", "a_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")

    # setup bus
    capsule.post()

    assert capsule.bus["sx"] == 5
    assert capsule.bus["sz"] == 2

    # consume bus
    capsule.read()
    assert capsule.dissipated == 3

    # check agents
    assert capsule.agents[0].state.symbol == "a_2"
    assert capsule.agents[1].state.symbol == "b_2"
    assert capsule.agents[2].state.symbol == "b_2"
    assert capsule.agents[3].state.symbol == "c_2"


def capsule_testing_distribution_2():
    a = Machine(name="A")

    a.add_state("a_1", flux=[Flow("sx"), Flow("sx"), Flow("sx")])
    a.add_state("a_2", flux=[Flow("sz")])

    a.add_flow("a_1", "a_2", flow=Flow("sz"))
    a.add_flow("a_2", "a_1", flow=Flow("sx"))

    b = Machine(name="B")

    b.add_state("b_1", flux=[Flow("sz")])
    b.add_state("b_2", flux=[Flow("sx")])

    b.add_flow("b_1", "b_2", flow=Flow("sx"))
    b.add_flow("b_2", "b_1", flow=Flow("sz"))

    c = Machine(name="C")

    c.add_state("c_1")
    c.add_state("c_2")

    c.add_flow("c_1", "c_2", flow=Flow("sx"))
    c.add_flow("c_2", "c_1", flow=Flow("sz"))

    bath = Bath(grid=1, machines=[a, b, c])
    capsule = bath.get_cell([0, 0])

    bath.instantiate(numpy.array([0, 0]), "A", "a_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")

    # setup bus
    bath.run()

    assert capsule.hang["b_1"] == 1
    assert capsule.hang["c_1"] == 1
    assert capsule.dissipated == 0

    states = {s: len([a for a in capsule.agents if a.state.symbol == s])
              for s in ["a_1", "a_2", "b_1", "b_2", "c_1", "c_2"]}

    assert states["a_1"] == 0
    assert states["a_2"] == 1
    assert states["b_1"] == 1
    assert states["b_2"] == 2
    assert states["c_1"] == 1
    assert states["c_2"] == 1


def capsule_testing_distribution_3():
    a = Machine(name="A")

    a.add_state("a_1", flux=[Flow("sx"), Flow("sx"), Flow("sx")])
    a.add_state("a_2", flux=[Flow("sz")])

    a.add_flow("a_1", "a_2", flow=Flow("sz"))
    a.add_flow("a_2", "a_1", flow=Flow("sx"))

    b = Machine(name="B")

    b.add_state("b_1", flux=[Flow("sz")])
    b.add_state("b_2", flux=[Flow("sx")])

    b.add_flow("b_1", "b_2", flow=Flow("sx"))
    b.add_flow("b_2", "b_1", flow=Flow("sz"))

    c = Machine(name="C")

    c.add_state("c_1")
    c.add_state("c_2")

    c.add_flow("c_1", "c_2", flow=Flow("sx"))
    c.add_flow("c_2", "c_1", flow=Flow("sz"))

    bath = Bath(grid=1, machines=[a, b, c])
    capsule = bath.get_cell([0, 0])

    bath.instantiate(numpy.array([0, 0]), "A", "a_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")
    bath.instantiate(numpy.array([0, 0]), "C", "c_1")

    # setup bus
    bath.run()

    assert capsule.hang["b_1"] == 2
    assert capsule.hang["c_1"] == 2
    assert capsule.dissipated == 0

    assert capsule.state["a_1"] == 0
    assert capsule.state["a_2"] == 1
    assert capsule.state["b_1"] == 2
    assert capsule.state["b_2"] == 1
    assert capsule.state["c_1"] == 2
    assert capsule.state["c_2"] == 1


def capsule_testing_distribution_yield():
    a = Machine(name="A")

    a.add_state("a_1", flux=[Flow("sx"), Nu("b_1")])
    a.add_state("a_2", flux=[Flow("sz"), Flow("sx")])

    a.add_flow("a_1", "a_2", flow=Flow("sz"))
    a.add_flow("a_2", "a_1", flow=Flow("sx"))

    b = Machine(name="B")

    b.add_state("b_1", flux=[Flow("sz")])
    b.add_state("b_2", flux=[Flow("sx")])

    b.add_flow("b_1", "b_2", flow=Flow("sx"))
    b.add_flow("b_2", "b_1", flow=Flow("sz"))

    bath = Bath(grid=1, machines=[a, b])
    capsule = bath.get_cell([0, 0])

    bath.instantiate(numpy.array([0, 0]), "A", "a_1")
    bath.instantiate(numpy.array([0, 0]), "B", "b_1")

    bath.run()

    assert capsule.state["a_1"] == 0
    assert capsule.state["a_2"] == 1
    assert capsule.state["b_1"] == 0
    assert capsule.state["b_2"] == 1

    bath.run()

    assert capsule.state["a_1"] == 1
    assert capsule.state["a_2"] == 0
    assert capsule.state["b_1"] == 1
    assert capsule.state["b_2"] == 1

    bath.run()

    assert capsule.state["a_1"] == 1  # sx -- sz
    assert capsule.state["a_2"] == 0  # sx, sz -- sx
    assert capsule.state["b_1"] == 0  # sz -- sx
    assert capsule.state["b_2"] == 2  # sx -- sz

    bath.run()

    assert capsule.state["a_1"] == 1
    assert capsule.state["a_2"] == 0
    assert capsule.state["b_1"] == 1
    assert capsule.state["b_2"] == 2

    bath.run()

    assert capsule.state["a_1"] == 0
    assert capsule.state["a_2"] == 1
    assert capsule.state["b_1"] == 1
    assert capsule.state["b_2"] == 3


def main():
    print("[+] testing")

    capsule_testing_cycle()

    capsule_testing_distribution_1()
    capsule_testing_distribution_2()
    capsule_testing_distribution_3()

    capsule_testing_distribution_yield()


main()
