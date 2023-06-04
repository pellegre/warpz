from warpz.space.box import *
from models.machines.automata import *


from enum import IntEnum
from bisect import bisect

import numpy
import time

# random state
random_state = numpy.random.RandomState()

# motility
MOTILITY = 0.00002

# cell radius
RADIUS = 0.005

# ===========
#
# Material unit
#
# ===========

WARPS = "warps"


class Unit(Capsule):
    def __init__(self, **kwargs):
        # material stack for this unit
        self.material = {WARPS: 0}
        super().__init__(**kwargs)

    def interact(self, signals):
        # physical assertion
        for chemokine in self.material:
            if self.material[chemokine] > 0 and chemokine != WARPS:
                self.material[chemokine] -= min(self.universe.cost, self.material[chemokine])

            # materials
            assert self.material[chemokine] >= 0

        # setup bus
        yield self.post()

        # consume it
        yield self.read()

    def get_warps(self, amount):
        # get glucose stack
        warps_stack = self.material[WARPS]
        if warps_stack >= amount:
            # give it as it is
            self.material[WARPS] -= amount
            return amount

        else:
            # give what it is
            self.material[WARPS] = 0
            return warps_stack


# ===========
#
# Medium (collection of material units)
#
# ===========

class Medium(Box):
    def __init__(self, warps, machines, grid=10, diffusion_rate=0.01, diffusion_period=20, chromo=None, **kwargs):
        # machines
        self.machines = {m.name: m for m in sorted(machines, key=lambda n: n.name)}

        # state chromo
        self.chromo = chromo if chromo is not None else dict()

        # initialize box
        super().__init__(grid=grid, cell=Unit, **kwargs)

        # total of warps
        self.warps, self.warps_price = warps, 0

        # diffusion parameters
        self.diffusion_rate, self.diffusion_period = diffusion_rate, diffusion_period

        # food and energy release
        self.mesh = numpy.zeros(self._dimensions * (self._grid,), float)

        # initial time
        self.tic, self.toc = time.process_time(), time.process_time()

        # cost function
        self.clock_time, self.cost, self.accumulated_cost = 0, 1, 0

    def instantiate(self, position, machine, state):
        # add agent
        agent = Cellulata(position=position, machine=self.machines[machine], state=state)
        self.put(agent)

    def put(self, child):
        # put child on universe
        super().put(child)

        if child.get_cell().material[WARPS] > 0:
            # take warps
            child.get_cell().material[WARPS] -= child.warps

            # adjust warps
            remain = child.get_cell().material[WARPS]
            if remain < 0:
                child.warps += remain
                # fix up level
                child.get_cell().material[WARPS] = 0
        else:
            self.warps -= child.warps

    def uniform_distribution(self, filled=True):
        # get cells
        if filled:
            children = self.get_children(condition=lambda c: len(c.get_children()) > 0)
        else:
            children = self.get_children()

        # distribute warps
        if len(children):
            each = self.warps // len(children)

            # distribute energy
            for cell in self.get_children():
                # distribute warps to the cell
                amount = min(each, self.warps)
                self.warps -= amount
                cell.material[WARPS] += amount

        # balance checking
        assert self.warps >= 0

    def update_cost(self):
        # initial time
        self.toc = time.process_time()

        # current delta
        delta = self.toc - self.tic

        # update clock delta average
        self.clock_time += delta

        # get average
        clock_delta = self.clock_time / (self.get_time() + 1)

        # update last clock
        self.tic = self.toc

        # update cost
        self.cost = max(0, min(7, int(delta / clock_delta) + 1))

        # accumulated cost
        self.accumulated_cost += self.cost

    def diffuse(self):
        # distribute back energy
        for cell in self.get_children():

            # get neighboring cells
            if self.boundary is Box.Boundary.CLOSED:
                neighbors = self.get_closed_neighbors(cell)
            else:
                neighbors = self.get_periodic_neighbors(cell)

            # distribute chemokines
            for chemokine in filter(lambda c: cell.material[c] > 0, cell.material):
                for each in filter(lambda c: c.material[chemokine] < cell.material[chemokine], neighbors):
                    # material delta
                    delta_material = cell.material[chemokine] - each.material[chemokine]

                    # get diffusive food energy
                    amount = min(cell.material[chemokine], max(1, int(self.diffusion_rate * delta_material)))
                    cell.material[chemokine] -= amount
                    each.material[chemokine] += amount

    def get_mesh(self):
        # mesh
        for each in self.get_children(condition=lambda c: isinstance(c, Unit)):
            self.mesh[each.index[1], each.index[0]] = math.log(each.material[WARPS] + 1)

        # get light mesh
        return self.mesh

    def __call__(self, signals):
        # diffuse
        if self.get_time() % self.diffusion_period == 0:
            self.diffuse()

        # update simulation cost
        self.update_cost()

        # periodic distribution of warps
        if self.get_time() % 15 == 0:
            self.uniform_distribution()


class Cellulata(Ball):
    # cellulata's color
    RELEASING_COLOR = 0.20
    GREEDY_COLOR = 0.80

    def __init__(self, machine, state, warps=120, **kwargs):
        # machine and state
        self.machine = machine

        # current state
        self.state = self.machine.symbols[state]

        # cellulata motility
        self.motility = MOTILITY * 3

        # direction angle
        self.direction = self.get_random_direction()

        # cellula warps
        self.warps = warps

        # born time
        self.born_time = None

        # apoptosis flag (surgery)
        self.apoptosis = False

        # inertial counter
        self.movement_counter = 0

        # greediness
        self.greediness = numpy.random.randint(1, 5)
        self.inertia = numpy.random.randint(2, 10)

        # agent instantiation
        radius = RADIUS

        # generated signals
        self.posting = self.get_posting_signals()
        self.listening = self.get_listening_signals()

        # default chromo
        self.color = 0.80

        # initialize agent
        super().__init__(radius=radius, **kwargs)

    def set_state(self, state):
        # current state
        self.state = self.machine.symbols[state]

        # generated signals
        self.posting = self.get_posting_signals()
        self.listening = self.get_listening_signals()

    def get_posting_signals(self):
        if self.state.flux is not None:
            return self.state.flux.flows

        return list()

    def get_listening_signals(self):
        return [flow.symbol for target in self.machine.output_flux[self.state]
                for flow in self.machine.output_flux[self.state][target]]

    @staticmethod
    def get_random_direction():
        # random direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def transport(self, units=1):
        # inertia
        if self.movement_counter >= self.inertia:
            # set new direction
            self.direction = self.get_random_direction()

            # reset inertial counter
            self.movement_counter = 0

        # tune up motility
        yield Transport(units * self.motility, self.direction)

        # count movement
        self.movement_counter += 1

    def consume(self):
        # update chromo
        if self.state.symbol in self.get_universe().chromo:
            self.color = self.get_universe().chromo[self.state.symbol]

        # get current cell
        cell = self.get_cell()

        # request warps
        warps = self.get_universe().cost + self.greediness
        self.warps += cell.get_warps(warps)

    def gone(self):
        # put back remaining warps
        if self.warps > 0:
            self.get_cell().warps += self.warps

        # cellula is done
        yield Gone()

    def done(self):
        # setup born time
        current = self.get_universe().get_time()
        if self.born_time is None:
            self.born_time = current

        # grace period on start
        return self.apoptosis or self.warps == 0

    def metabolize(self):
        if self.warps > 0:
            # metabolize warps
            metabolized = min(self.warps, self.get_universe().cost ** 2)

            # balance warps
            self.warps -= metabolized
            self.get_universe().warps += metabolized

    def replicate(self):
        # release the child my friend
        if self.warps >= 2 * 120:
            # balance warps
            born_warps = self.warps // 2
            self.warps -= born_warps

            # replicate
            yield Replicate(born=Cellulata(warps=born_warps, position=self.get_position(),
                                           machine=self.machine, state=self.state.symbol))

        # verify physical sanity
        assert self.warps >= 0

    def interact(self, signals):
        # interaction loop
        if not self.done():
            # consume
            self.consume()

            # metabolize
            self.metabolize()

            # replicate
            yield self.replicate()

            # move
            yield self.transport()
        else:
            # gone
            yield self.gone()

