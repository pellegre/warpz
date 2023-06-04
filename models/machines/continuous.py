from abc import ABC

import sympy
import copy

import pandas
import numpy

from scipy.integrate import solve_ivp
from pde import FieldCollection, PDEBase, PlotTracker, ScalarField, UnitGrid


class System:
    def __init__(self, machines):
        self.machines = machines

        # state rates
        self.states = set()
        self.state_rates = dict()

        # fluxes
        self.fluxes = dict()

        # machine state and flow rate
        for machine in self.machines:
            for each in machine.states:
                if each not in machine.final:
                    # store state
                    self.states.add(sympy.Symbol(each.symbol))

                    # yield and decay rate
                    nu, decay = sympy.Symbol("nu_" + each.symbol), sympy.Symbol("lambda_" + each.symbol)

                    # balance output flow
                    output_flux = sympy.Integer(0)
                    for target in machine.output_flux[each]:
                        for flow in machine.output_flux[each][target]:
                            output_flow = sympy.Symbol(flow.symbol) * flow.value * sympy.Symbol(each.symbol)
                            output_flux = (output_flux + output_flow)

                            if flow.flux is not None:
                                for chain_flow in flow.flux.flows:
                                    symbol = chain_flow.symbol

                                    if symbol not in self.fluxes:
                                        self.fluxes[symbol] = sympy.Integer(0)

                                    self.fluxes[symbol] = (self.fluxes[symbol] +
                                                           chain_flow.value * output_flow)

                    # balance input flow
                    input_flux = sympy.Integer(0)
                    for source in machine.input_flux[each]:
                        for flow in machine.input_flux[each][source]:
                            input_flow = sympy.Symbol(flow.symbol) * flow.value * sympy.Symbol(source.symbol)
                            input_flux = (input_flux + input_flow)

                    if each.flux is not None:
                        for flow in each.flux.flows:
                            symbol = flow.symbol
                            if symbol not in self.fluxes:
                                self.fluxes[symbol] = sympy.Integer(0)

                            self.fluxes[symbol] = (self.fluxes[symbol] + flow.value * sympy.Symbol(each.symbol))

                    # initial expressions
                    self.state_rates["rate_" + each.symbol] = (nu - decay - output_flux + input_flux)

        for each in self.fluxes:
            print("+++", each, "=", self.fluxes[each])

        for each in self.state_rates:
            print("+++", each, "=", self.state_rates[each])

        self.rates = self._solve_rates(self.state_rates, self.fluxes)

    def balance(self, state, external_fluxes=None):
        fluxes = copy.deepcopy(self.fluxes)
        if external_fluxes is not None:
            for flux in external_fluxes:
                if flux not in fluxes:
                    fluxes[flux] = external_fluxes[flux]
                else:
                    fluxes[flux] = (fluxes[flux] + external_fluxes[flux])

        else:
            external_fluxes = list()

        for r in self.rates:
            print(r, "=", self.rates[r])

        print("-")

        for machine in self.machines:
            for each in machine.states:
                if each not in machine.final:
                    nu, decay = sympy.Symbol("nu_" + each.symbol), sympy.Symbol("lambda_" + each.symbol)

                    if nu not in self.rates and str(nu) not in external_fluxes:
                        fluxes["nu_" + each.symbol] = sympy.Integer(0)

                    if decay not in self.rates and str(decay) not in external_fluxes:
                        fluxes["lambda_" + each.symbol] = sympy.Integer(0)

        solved = self._solve_rates(self.state_rates, fluxes)
        rates = {r: solved[r] for r in filter(lambda q: str(q) in self.state_rates, solved)}

        undefined = set()
        for each in rates:
            undefined.update(rates[each].free_symbols)

        undefined = undefined.difference(self.states)

        if len(undefined) > 0:
            raise RuntimeError("undefined flows " + str(undefined))

        initial = {s: state[str(s)] if str(s) in state else 0 for s in self.states}

        machines = {each.name: [state.symbol for state in each.states if state not in each.final]
                    for each in self.machines}

        return initial, rates, machines

    @staticmethod
    def _solve_rates(state_rates, fluxes):
        rates, symbols = list(), list()

        for each in state_rates:
            symbol = sympy.Symbol(each)

            # state rate
            rates.append(sympy.Eq(symbol, state_rates[each]))
            symbols.append(symbol)

        for each in fluxes:
            symbol = sympy.Symbol(each)

            # flux rate
            rates.append(sympy.Eq(symbol, fluxes[each]))
            symbols.append(symbol)

        return sympy.solve(rates, symbols)


class Punctual:
    def __init__(self, machines, state=None, rates=None, time_span=numpy.linspace(0, 200, 400)):
        # state and rates
        self.state = state
        self.rates = rates
        self.machines = machines

        # time span
        self.time_span = time_span

        # rate pointer
        self.ptr = {i: s for i, s in enumerate(self.state)}
        self.dr_dt = [0] * len(self.ptr)

        for i in range(0, len(self.dr_dt)):
            self.dr_dt[i] = self.rates[sympy.Symbol("rate_" + str(self.ptr[i]))]

    def eval(self, t, r):
        states = {self.ptr[i]: r[i] for i in range(0, len(r))}
        dr_dt = [e.subs(states) for e in self.dr_dt]

        return dr_dt

    def solve(self):
        initial = [self.state[self.ptr[i]] for i in range(0, len(self.ptr))]
        sol_nar = solve_ivp(lambda t, y: self.eval(t, y), [self.time_span[0], self.time_span[-1]], initial,
                            t_eval=self.time_span)

        sol = pandas.DataFrame(numpy.transpose(numpy.array([sol_nar.t] +
                                                           [sol_nar.y[i] for i in range(0, len(self.ptr))])),
                               columns=["time"] + [str(self.ptr[i]) for i in range(0, len(self.ptr))])

        # time index
        sol = sol.set_index("time")

        for each in self.machines:
            sol[each] = 0
            for state in self.machines[each]:
                sol[each] += sol[state]

        return sol


class Diffusion(PDEBase, ABC):
    def __init__(self, state, rates, machines, diffusion):
        # state and rates
        self.rates = rates
        self.machines = machines

        # boundary condition
        self.bc = "periodic"

        # diffusion terms
        self.diffusion = dict()
        for m in self.machines:

            if m not in diffusion:
                raise RuntimeError("diffusion is not set for machine " + m)

            for each in self.machines[m]:
                self.diffusion[each] = diffusion[m]

        # rate pointer
        self.ptr = {i: s for i, s in enumerate(state)}
        self.dr_dt = [0] * len(self.ptr)

        for i in range(0, len(self.dr_dt)):
            self.dr_dt[i] = self.rates[sympy.Symbol("rate_" + str(self.ptr[i]))]

        self.grid = UnitGrid([60, 60], periodic=[True, True])
        self.state = FieldCollection([ScalarField(self.grid, 0) for _ in state])

        for i in self.ptr:
            self.state[i].data[0:5, :] = state[self.ptr[i]]

    def evolution_rate(self, r, t=0):
        states = {self.ptr[i]: r[i] for i in range(0, len(r))}
        print(states)
        dr_dt = [r[i].laplace(self.bc) + e.subs(states) for i, e in enumerate(self.dr_dt)]

        return FieldCollection(dr_dt)

    def track(self):
        tracker = PlotTracker(interval=1)
        sol = self.solve(self.state, t_range=50000, dt=1e-2, tracker=["progress", tracker])
