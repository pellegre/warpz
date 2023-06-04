from warpz.space.box import *

from models.machines.network import *

import math
import random
import sympy
import pandas
import numpy

from scipy.integrate import solve_ivp

# number of attempts to reduce the leak
NU = 2000
LINK_TIME = 200


class Automata(Particle):
    def __init__(self, machine, state, q, **kwargs):
        # machine reference
        self.machine = machine

        # information capacity
        self.q = q

        # set state
        self.state, self.posting, self.listening = None, None, None

        # set state
        self.set_state(state)

        # stall state
        self.stall = False

        # gone
        self.gone = False

        super().__init__(**kwargs)

    def get_posting_signals(self):
        if self.state.flux is not None:
            return self.state.flux.flows

        return list()

    def get_listening_signals(self):
        return [flow.symbol for target in self.machine.output_flux[self.state]
                for flow in self.machine.output_flux[self.state][target]]

    def set_state(self, state):
        # current state
        self.state = self.machine.symbols[state]

    def setup(self):
        # reset stalled state
        self.stall = False

        # generate signals
        posting, self.posting = self.get_posting_signals(), list()
        for each in posting:
            # check capacity
            if self.q > 0:
                self.posting.append(each)
            else:
                # agent is stalled
                self.stall = True

        # set listening signals
        self.listening = self.get_listening_signals()

    def interact(self, signals):
        if self.gone:
            yield Gone()


class Capsule(Cell):
    def __init__(self, **kwargs):
        # initialize machines
        self.machines = kwargs["universe"].machines
        self.agents, self.bus, self.listening = list(), dict(), dict()

        # state mapping
        self.state_mapping = {s.symbol: m for m in self.machines for s in self.machines[m].states}

        # collision and dissipation
        self.dissipated, self.hang = 0, {m: 0 for m in self.state_mapping}

        # capsule state
        self.state = self.get_state()

        # flows
        self.flows = dict()

        # setup flows
        for m in self.machines:
            for flow in self.machines[m].flows:
                # setup flow
                if flow.symbol not in self.flows:
                    self.flows[flow.symbol] = list()

                # accumulate flow
                self.flows[flow.symbol].append(flow)

        # produced and absorbed flow
        self.produced, self.absorbed = set(), set()

        # collect flows
        for f in self.flows:
            for flow in self.flows[f]:
                # collect produced flows
                if flow.target is None:
                    # accumulate flow
                    self.produced.add(flow.source)
                else:
                    # accumulate flow
                    self.absorbed.add(flow.source)

        super().__init__(limit=50, **kwargs)

    @staticmethod
    def within_three_sigma_round(n):
        if n - math.floor(n) <= 0.5:
            return int(math.floor(n))
        return int(math.ceil(n))

    def get_state(self):
        return {s: len([a for a in self.agents if a.state.symbol == s]) for s in self.state_mapping}

    def leak(self):
        produced, absorbed = 0, 0
        for agent in self.agents:
            # produced signals
            if agent.state in self.produced:
                produced += agent.q

            # absorbed signals
            if agent.state in self.absorbed:
                absorbed += 1

        # leak
        return produced - absorbed

    def delta_leak_after_switch(self, i, j):
        # get leak
        leak = self.leak()

        if self.agents[i].q > 0:
            # take unit
            if self.agents[i].state in self.produced:
                leak -= 1

            # take unit
            if self.agents[j].state in self.produced:
                leak += 1

            # sanity
            assert self.agents[i].q >= 0
            assert self.agents[j].q >= 0

        # after switch
        return leak - self.leak()

    def replicate(self):
        for agent in self.agents:
            # set posting signals
            for signal in agent.posting:
                if isinstance(signal, Nu):
                    # yield agent
                    for i in range(0, int(signal.value)):
                        if agent.q > 0:
                            # get machine
                            machine = self.machines[self.state_mapping[signal.state.symbol]]

                            # setup q
                            q = agent.q // 2
                            agent.q = (agent.q - q)

                            # replicate
                            self.universe.put(Automata(position=agent.get_position(), machine=machine,
                                                       state=signal.state.symbol, q=q))

    def metropolis(self):
        # choose random agents
        i, j = random.randint(0, len(self.agents) - 1), random.randint(0, len(self.agents) - 1)

        # check capacity
        if self.agents[i].q > 0:
            # delta
            delta = self.delta_leak_after_switch(i, j)

            if delta > 0:
                prob = math.exp(-delta / self.universe.information)
                if random.uniform(0, 1) < prob:
                    # switch
                    self.agents[j].q += 1
                    self.agents[i].q -= 1
            elif delta < 0:
                # switch
                self.agents[j].q += 1
                self.agents[i].q -= 1

    def thermal(self):
        # setup agents
        self.agents = self.get_children()

        # perform metropolis
        count, leak = 0, self.leak()
        for n in range(0, NU):
            # perform thermalization
            self.metropolis()

            # get leak
            leak_after = self.leak()
            if leak_after == leak:
                # break on times
                count += 1
                if count == LINK_TIME:
                    break
            else:
                # reset
                count = 0

        # post signals
        for agent in self.agents:
            # setup signals
            agent.setup()

    def post(self):
        # post signals
        for agent in self.agents:
            # set posting signals
            for signal in agent.posting:
                if not isinstance(signal, Nu):
                    # setup listening
                    if signal.symbol not in self.bus:
                        self.bus[signal.symbol] = int(agent.q * signal.value)
                    else:
                        self.bus[signal.symbol] += int(agent.q * signal.value)

            # set listening signals
            for signal in agent.listening:
                # setup listening
                if signal not in self.listening:
                    self.listening[signal] = set()

                # track agent
                self.listening[signal].add(agent)

            # sanity
            assert agent.q >= 0

    def read(self):
        # agent's queue
        queue = {a for a in self.agents}

        # signal distribution
        for signal in self.listening:
            # get listening agents
            agents = {a for a in self.listening[signal] if a in queue}

            # look up bus
            if signal in self.bus and self.bus[signal] > 0:
                if len(agents) <= self.bus[signal]:
                    # pick up all signals
                    self.bus[signal] -= len(agents)
                    queue = queue.difference(agents)

                    # move state
                    for a in agents:
                        # set state
                        a.set_state(a.machine.transition[a.state][signal].symbol)

                else:
                    # count signals desires
                    desires = {m: len([a for a in agents if a.state.symbol == m]) for m in self.state_mapping}

                    # setup democratic distribution
                    democratic = {m: self.within_three_sigma_round(self.bus[signal] * (desires[m] / len(agents)))
                                  for m in desires}

                    # total signals distributed
                    total = sum([democratic[m] for m in democratic])
                    if total == self.bus[signal] + 1:
                        # make a choice
                        choice = random.choice(list(filter(lambda a: democratic[a] > 0, democratic.keys())))
                        democratic[choice] -= 1

                        # recalculate total
                        total = sum([democratic[m] for m in democratic])

                    # sanity
                    assert total == self.bus[signal] or total == self.bus[signal] - 1

                    # hanged agents
                    self.hang = {m: self.hang[m] + (desires[m] - democratic[m]) for m in self.hang}

                    # distribute signal
                    for a in agents:
                        if democratic[a.state.symbol] > 0:
                            # move forward
                            democratic[a.state.symbol] -= 1
                            a.set_state(a.machine.transition[a.state][signal].symbol)

                            # remove from bus
                            self.bus[signal] -= 1

                            # take out agent
                            queue.remove(a)

        # count collisions
        for signal in self.bus:
            self.dissipated += self.bus[signal]

        # update capsule state
        self.state = self.get_state()

        # clear listening queue
        self.listening.clear()
        self.bus.clear()

    def run(self):
        # thermal run
        self.thermal()

        # setup bus
        yield self.post()

        # consume it
        yield self.read()

        # replicate agents
        yield self.replicate()

    def show(self):
        print("[+] state", self.state)
        for agent in self.agents:
            print("[-]  ", agent.machine.name, "at", agent.state, " --- ", agent.listening)

    def interact(self, signals):
        # run
        yield self.run()


class Bath(Box):
    def __init__(self, grid, machines, information, q=100, **kwargs):
        # machines
        self.machines = {m.name: m for m in sorted(machines, key=lambda n: n.name)}

        # information and bandwidth
        self.information = information
        self.q = q

        # initialize box
        super().__init__(grid=grid, cell=Capsule, **kwargs)

    def instantiate(self, position, machine, state):
        # add agent
        agent = Automata(position=position, machine=self.machines[machine], state=state, q=self.q)
        self.put(agent)


class Transport:
    def __init__(self, machines):
        # machines
        self.machines = {m.name: m for m in sorted(machines, key=lambda n: n.name)}
        self.output_flux = dict()

        # state mapping
        self.states = {s.symbol: s for m in self.machines for s in self.machines[m].states}

        for m in self.machines:
            self.output_flux |= self.machines[m].output_flux

        # flows
        self.flows = dict()

        # setup flows
        for m in self.machines:
            for flow in self.machines[m].flows:
                # setup flow
                if flow.symbol not in self.flows:
                    self.flows[flow.symbol] = list()

                # accumulate flow
                self.flows[flow.symbol].append(flow)

        # produced and absorbed flow
        produced, absorbed = dict(), dict()

        # collect flows
        for f in self.flows:
            for flow in self.flows[f]:
                # collect produced flows
                if flow.target is None:
                    # setup produced flow
                    if f not in produced:
                        produced[f] = list()

                    # accumulate flow
                    produced[f].append(flow)
                else:
                    # collect absorbed flows
                    if f not in absorbed:
                        absorbed[f] = list()

                    # accumulate flow
                    absorbed[f].append(flow)

        self.produced, self.nu_flow = dict(), dict()
        for flow in produced:
            # initialize flow
            self.produced[flow] = sympy.Integer(0)

            for each in produced[flow]:
                if isinstance(each, Nu):
                    # rate (same for each state)
                    state = sympy.Symbol("N_" + each.source.symbol)
                    q = sympy.Symbol("nu_" + each.state.symbol)

                    # produced flow
                    self.nu_flow[each.source.symbol] = (self.produced[flow] + q * state)
                else:
                    # rate (same for each state)
                    state = sympy.Symbol("N_" + each.source.symbol)
                    q = sympy.Symbol("r_" + flow)

                    # produced flow
                    self.produced[flow] = (self.produced[flow] + (q * state))

        self.absorbed, self.absorbed_per_state, self.scatter_per_state = dict(), dict(), dict()
        for flow in absorbed:
            # initialize flow
            self.absorbed[flow] = sympy.Integer(0)

            # total absorption
            states = sympy.Integer(0)

            # rate (same for each state)
            for each in absorbed[flow]:
                # state
                state = sympy.Symbol("N_" + each.source.symbol)

                # accumulate state
                states = (states + state)

                # absorbed flow
                self.absorbed[flow] = (self.absorbed[flow] + state)

            # rate (same for each state)
            for each in absorbed[flow]:
                # collect state
                state = each.source.symbol
                if state not in self.absorbed_per_state:
                    self.absorbed_per_state[state] = sympy.Integer(0)

                # distribute bandwidth
                symbol = sympy.Symbol("N_" + state)
                q = sympy.Symbol("a_" + flow)

                # collect absorbed
                absorbed_in_state = sympy.Min(q * symbol, (symbol * self.produced[flow]) / states)
                self.absorbed_per_state[state] = (self.absorbed_per_state[state] + absorbed_in_state)

                # collect scatter
                target = each.target.symbol
                if target not in self.scatter_per_state:
                    self.scatter_per_state[target] = sympy.Integer(0)

                # collect absorbed
                self.scatter_per_state[target] = (self.scatter_per_state[target] + absorbed_in_state)

        self.flow = dict()

        for state in self.states:
            self.flow[state] = sympy.Integer(0)

            # production (nu)
            nu = sympy.Symbol("nu_" + state)
            self.flow[state] = (self.flow[state] + nu * sympy.Symbol("N_" + state))

            # scatter
            if state in self.scatter_per_state:
                self.flow[state] = (self.flow[state] + self.scatter_per_state[state])

            # absorption
            if state in self.absorbed_per_state:
                self.flow[state] = (self.flow[state] - self.absorbed_per_state[state])

        # set rates (production and absorption)
        self.production = {"r_" + flow: 1 for flow in self.flows}
        self.absorption = {"a_" + flow: 1 for flow in self.flows}
        self.nu = {"nu_" + state: 0 for state in self.states}

        # initial conditions
        self.initial = {state: 0 for state in self.states}

    def show(self):
        print("[+] initial condition")
        for each in self.initial:
            print("   [#]", each, "=", self.initial[each])

        print("[+] signal rates")
        for p, a, nu in zip(self.production, self.absorption, self.nu):
            print("   [#] ", p, "=", self.production[p], ",", a, "=", self.absorption[a], ",", nu, "=", self.nu[nu])

        print("[+] absorbed (per state)")
        for each in self.absorbed_per_state:
            print("   [#]", each, "=", self.absorbed_per_state[each])

        print("[+] scatter (per state)")
        for each in self.scatter_per_state:
            print("   [#]", each, "=", self.scatter_per_state[each])

        print("[+] production (per state)")
        for each in self.nu:
            print("   [#]", each, "=", self.nu[each])

        print("[+] flow balance")
        for each in self.flow:
            print("   [#]", each, "=", self.flow[each])


class Punctual(Transport):
    def __init__(self, machines, initial=None, rates=None, time_span=numpy.linspace(0, 20, 100)):
        # initialize
        super(Punctual, self).__init__(machines=machines)

        # set initial condition
        if initial is not None:
            for each in initial:
                if each not in self.initial:
                    raise RuntimeError("state not found " + str(each))
                else:
                    self.initial[each] = initial[each]

        # set initial condition
        if rates is not None:
            for each in rates:
                if each in self.absorption:
                    self.absorption[each] = rates[each]
                elif each in self.production:
                    self.production[each] = rates[each]
                elif each in self.nu:
                    self.nu[each] = rates[each]
                else:
                    raise RuntimeError("rate not found " + str(each))

        # time span
        self.time_span = time_span

        # rate pointer
        self.ptr = {i: s for i, s in enumerate(self.states)}
        self.dr_dt = [0] * len(self.ptr)

        # rate equations
        for i in range(0, len(self.dr_dt)):
            self.dr_dt[i] = self.flow[self.ptr[i]].subs(self.absorption | self.production | self.nu)

        # show state
        self.show()
        print("[+] rate equations")
        for i in range(0, len(self.dr_dt)):
            print("   [#]", self.ptr[i], "=", self.dr_dt[i])

    def eval(self, t, r):
        states = {"N_" + self.ptr[i]: r[i] for i in range(0, len(r))}
        dr_dt = [e.subs(states) for e in self.dr_dt]

        return dr_dt

    def solve(self):
        initial = [self.initial[self.ptr[i]] for i in range(0, len(self.ptr))]
        sol_nar = solve_ivp(lambda t, y: self.eval(t, y), [self.time_span[0], self.time_span[-1]], initial,
                            t_eval=self.time_span)

        sol = pandas.DataFrame(numpy.transpose(numpy.array([sol_nar.t] +
                                                           [sol_nar.y[i] for i in range(0, len(self.ptr))])),
                               columns=["time"] + [str(self.ptr[i]) for i in range(0, len(self.ptr))])

        # time index
        sol = sol.set_index("time")

        for each in self.machines:
            sol[each] = 0
            for state in self.machines[each].states:
                sol[each] += sol[state.symbol]

        return sol
