import networkx

from collections.abc import Iterable


class Flux:
    def __init__(self, flows):
        if flows is None:
            self.flows = list()
        elif isinstance(flows, Flow):
            self.flows = [flows]
        elif isinstance(flows, Iterable):
            self.flows = list(flows)
        else:
            raise RuntimeError("unsupported flow " + str(flows))

    def add_flow(self, flows):
        if isinstance(flows, Flow):
            self.flows.append(flows)
        elif isinstance(flows, Iterable):
            self.flows += list(flows)
        else:
            raise RuntimeError("unsupported flow " + str(flows))

    def __str__(self):
        return "[%s]" % ", ".join(map(str, self.flows))


class Flow:
    def __init__(self, symbol, flux=None, value=1.0):
        self.symbol = symbol
        self.value = value

        if isinstance(flux, Flux):
            self.flux = flux
        elif isinstance(flux, Flow) or isinstance(flux, list):
            self.flux = Flux(flows=flux)
        else:
            if flux is not None:
                raise RuntimeError("unsupported flux " + str(flux))
            self.flux = None

        self.source, self.target = None, None

    def __hash__(self):
        return self.symbol.__hash__()

    def __str__(self):
        if self.flux is None:
            data = "flow(" + self.symbol + ") ; " + self.source.symbol

            if self.target is not None:
                data += " -> " + self.target.symbol

            return data
        else:
            return "flow(" + self.symbol + " -> " + str(self.flux) + ")"


class State:
    def __init__(self, symbol, flux=None):
        self.symbol = symbol

        if isinstance(flux, Flux):
            self.flux = flux
        elif isinstance(flux, Flow) or isinstance(flux, list):
            if isinstance(flux, Flow):
                flux.source = self
            else:
                for each in flux:
                    each.source = self

            self.flux = Flux(flows=flux)
        else:
            if flux is not None:
                raise RuntimeError("unsupported flux " + str(flux))
            self.flux = None

    def __eq__(self, other):
        if not isinstance(other, State):
            return False

        return self.symbol == other.symbol

    def __hash__(self):
        return self.symbol.__hash__()

    def __str__(self):
        if self.flux is None:
            return "state(" + self.symbol + ")"
        else:
            return "state(" + self.symbol + " -> " + str(self.flux) + ")"


class Source(Flow):
    def __init__(self, symbol, state, value=1.0):
        if not isinstance(state, State):
            state = State(state)

        # state at which the flow is
        self.state = state
        super().__init__(symbol=symbol + "_" + state.symbol, value=value)


class Nu(Source):
    def __init__(self, state, value=1.0):
        super().__init__(symbol="nu", state=state, value=value)


class Lambda(Source):
    def __init__(self, state, value=1.0):
        super().__init__(symbol="lambda", state=state, value=value)


class Machine:
    def __init__(self, name):
        # machine network
        self.name = name
        self.g = networkx.DiGraph()

        # machine states
        self.states = set()
        self.final = set()
        self.flows = set()

        # symbols used by the machine
        self.symbols = dict()

        # state fluxes
        self.input_flux = dict()
        self.output_flux = dict()
        self.transition = dict()

    def __str__(self):
        data = "[machine " + self.name + "]\n"
        data += "  [= states]\n"

        for each in self.states:
            data += "  - " + str(each) + "\n"

        data += "  [= flow]\n"
        for each in self.output_flux:
            if len(self.output_flux[each]) > 0:
                for state in self.output_flux[each]:
                    data += "  - " + each.symbol + " ->"
                    data += " " + str(state.symbol) + \
                            " = " + "[%s]" % ", ".join(map(str, self.output_flux[each][state]))

                data += "\n"

        return data

    def add_state(self, state, flux=None, final=False):
        assert not isinstance(flux, Flux)

        if not isinstance(state, State):
            state = State(state, flux=flux)

        if state in self.states:
            raise RuntimeError("state already in machine " + str(state))

        # store state
        self.states.add(state)
        self.symbols[state.symbol] = state

        # is a final state
        if final:
            self.final.add(state)

        # store flow
        if state.flux is not None:
            for flow in state.flux.flows:
                self.flows.add(flow)

        # initialize fluxes
        self.input_flux[state] = dict()
        self.output_flux[state] = dict()
        self.transition[state] = dict()

        # state instance in the network
        self.g.add_node(state.symbol)

    def add_flow(self, source, target, flow):
        if not isinstance(source, State):
            source = State(source)

        if not isinstance(target, State):
            target = State(target)

        if source not in self.states:
            raise RuntimeError("state not in machine " + str(source))

        if target not in self.states:
            raise RuntimeError("state not in machine " + str(target))

        # set source and target
        flow.source, flow.target = source, target

        # input flow
        if source not in self.input_flux[target]:
            self.transition[source][flow.symbol] = target
            self.input_flux[target][source] = [flow]
        else:
            self.transition[source][flow.symbol] = target
            self.input_flux[target][source].append(flow)

        # output flow
        if target not in self.output_flux[source]:
            self.output_flux[source][target] = [flow]
        else:
            self.output_flux[source][target].append(flow)

        # store flow
        self.flows.add(flow)
        self.symbols[flow.symbol] = flow

        # store flow
        if flow.flux is not None:
            for flow in flow.flux.flows:
                self.flows.add(flow)

        # set network edge
        edge = self.g.get_edge_data(source, target)
        if edge is not None:
            edge["flow"].append(flow.symbol)
        else:
            self.g.add_edge(source.symbol, target.symbol, flow=[flow.symbol])
