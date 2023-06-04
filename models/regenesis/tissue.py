from warpz.space.box import *
from models.warper.source import *

from enum import IntEnum

import math
import numpy
import time
import copy

# random state
random_state = numpy.random.RandomState()

# floating point epsilon
EPSILON = 10E-07

# max value of a gene
GENE_CAPACITY = 256


# gene traits
class Trait(IntEnum):
    MOTILITY = 0
    GREEDINESS = 1
    INERTIA = 2
    BORN = 3
    MUTATION = 4
    SIGNALING = 5
    AMPLIFICATION = 6
    TOTAL = 7


class Food(Cell):
    def __init__(self, **kwargs):
        # initial stack of energy
        self.warps, self.initial = 0, 0
        self.chalones = 0
        super().__init__(limit=math.inf, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.chalones > 0:
            self.chalones -= 1


class Dish(Box):
    def __init__(self, warps_seed, grid=10, diffusion_rate=0.01,
                 diffusion_period=20, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Food, **kwargs)

        # motility
        self.motility = 1 / (500 * grid)

        # amount of warps initially distributed as food (i.e. seed)
        self.warps_seed = warps_seed

        # released warps by bugs
        self.warps = 0

        # food per cell
        children, self.grid = self.get_children(), grid
        cells = (len(children) - 4 * self.grid - 4 * (self.grid - 4))
        for each in children:
            if 2 <= each.index[0] < (grid - 2) and 2 <= each.index[1] < (grid - 2):
                each.warps, each.initial = warps_seed / cells, warps_seed / cells

        # initial time
        self.tic, self.toc = time.process_time(), time.process_time()
        # cost function
        self.clock_time, self.cost, self.accumulated_cost = 0, 1, 0

        # diffusion parameters
        self.diffusion_rate, self.diffusion_period = diffusion_rate, diffusion_period

        # food and energy release
        self.food = numpy.zeros(self._dimensions * (self._grid,), float)

    def get_mesh(self):
        # food mesh
        for each in self.get_children():
            self.food[each.index[1], each.index[0]] = math.log(each.chalones + 1)
        return self.food

    def get_cost(self):
        return self.cost ** 2

    def get_average_cost(self):
        return self.accumulated_cost / self.get_time()

    def put(self, child):
        # get warps for child
        self.warps_seed -= child.warps

        # put child on universe
        super().put(child)

        # adjust cell warping
        child.get_cell().warps -= child.warps

    def uniform_distribution(self):
        # get cells
        cells = self.get_children()
        each = self.warps / len(cells)

        # distribute energy
        for cell in self.get_children():
            amount = min(each, self.warps)
            self.warps -= amount
            cell.warps += amount

        # balance checking
        assert math.fabs(self.warps) >= -EPSILON
        self.warps = 0

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
        # self.cost = int(delta / clock_delta)
        self.cost = delta / clock_delta

        # accumulated cost
        self.accumulated_cost += self.cost

    def diffuse(self):
        # distribute back energy
        for cell in self.get_children():
            # get neighbors
            for each in cell.neighbors:
                if each.warps < cell.warps:
                    # energy delta
                    delta_energy = cell.warps - each.warps

                    # get diffusive food energy
                    amount = min(cell.warps, max(1, self.diffusion_rate * delta_energy))
                    cell.warps -= amount
                    each.warps += amount

    def __call__(self, *args, **kwargs):
        # update simulation cost
        self.update_cost()

        # diffusion
        if self.get_time() % self.diffusion_period == 0:
            self.diffuse()

        # redistribute energy
        if self.get_time() % 25 == 0:
            self.uniform_distribution()

# ===========
#
# signals
#
# ===========


# energy release signal
class Energy(Signal):
    def __init__(self, units=1, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, universe):
        # signal source
        who = self.get_source()

        # cost in warps
        warps = min(who.warps, self.units * universe.get_cost())

        # warps release to the universe
        universe.warps += warps

        # get warps from the bug
        who.warps -= warps
        max(0, who.warps)


# eat signal
class Eat(Signal):
    def __init__(self, amount, **kwargs):
        self.amount = amount
        super().__init__(**kwargs)

    def __call__(self, other):
        # eating energy
        amount = self.amount
        warps = min(other.warps, amount)

        # get eater
        eater = self.get_source()
        if eater.warps >= EPSILON:
            # energy balance
            other.warps -= warps
            max(0, other.warps)

            # get warps for eater
            eater.warps += warps

# ===========
#
# cell
#
# ===========


class Cellulata(Ball):
    def __init__(self, dish, gene, **kwargs):
        # bug time
        self.born_time = 0

        # gene
        self.gene = gene

        # energy
        self.dish = dish
        self.warps = self.gene.get_trait(Trait.BORN)

        # gene mutation
        mutability = self.gene.get_trait(Trait.MUTATION) / self.gene.get_capacity()
        if random_state.uniform(0, 1) < mutability:
            self.gene.mutate(random_state)

        # random direction
        self.direction = self.get_random_direction()

        # bug energy release
        self.energy_release = 0

        # inertia counter
        self.movement_counter = 0

        # apoptosis
        self.apoptosis = False

        # agent instantiation
        super().__init__(radius=self.radius, **kwargs)

    def set_direction(self, direction, priority=1):
        # inertia
        if self.movement_counter >= int(priority * self.gene.get_trait(Trait.INERTIA)):
            # set new direction
            self.direction = direction

            # reset inertial counter
            self.movement_counter = 0

    @staticmethod
    def get_random_direction():
        # random direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def release(self):
        # release energy
        yield Energy(units=self.energy_release, target=self.dish)

        # reset energy release
        self.energy_release = 0

    def transport(self, units=1):
        # move
        yield Transport(units * self.dish.motility, self.direction)

        # energy when moving
        self.energy_release += math.sqrt(units)

        # new time step
        self.movement_counter += 1

    def move(self):
        # attempt to move, after inertial property
        if self.movement_counter >= self.gene.get_trait(Trait.INERTIA):
            # random direction
            self.direction = self.get_random_direction()
            self.movement_counter = 0

        # bug transport
        yield self.transport(self.gene.get_trait(Trait.MOTILITY))

    def get_descendant(self):
        return Cellulata(gene=copy.deepcopy(self.gene), dish=self.dish)

    def reproduce(self):
        # release the child when enough energy has been harvested
        if self.warps >= 2 * self.gene.get_trait(Trait.BORN):
            descendant = self.get_descendant()
            if descendant is not None:
                assert isinstance(self, Stem)
                # new born child
                self.warps -= self.gene.get_trait(Trait.BORN)

                # replicate
                yield Replicate(born=descendant)

                # energy on all of us
                self.energy_release += 1

    def get_greediness(self):
        return self.gene.get_trait(Trait.GREEDINESS)

    def eat(self):
        # hungry condition
        if self.warps <= 3 * self.gene.get_trait(Trait.BORN):
            # bug eating greediness
            greediness = self.get_greediness()
            # warps I should eat
            warps = self.energy_release

            # eat food
            yield Eat(amount=greediness + warps, target=self.get_cell(), capacity=1)

    def done(self):
        # setup born time
        if not self.born_time:
            self.born_time = self.dish.get_time()
        return math.fabs(self.warps) <= EPSILON or self.apoptosis

    @staticmethod
    def signaling():
        # no signal
        yield None

    def gone(self):
        # agent is gone
        self.get_cell().warps += self.warps
        self.warps = 0

        # bug is done
        yield Gone()

    def interact(self, signals):
        if not self.done():
            # release signaling molecules
            yield self.signaling()

            # move my friend
            yield self.move()

            # mate and reproduce
            yield self.reproduce()
            yield self.eat()

            # energy release
            yield self.release()

        else:
            # bug is gone
            yield self.gone()


class Stem(Cellulata):
    def __init__(self, gene, signal=False, **kwargs):
        self.chromo, self.radius = 0.3, 10  # green
        self.signal, self.chalones = signal, 0

        # radius
        self.radius = 0.006

        super().__init__(gene=gene, **kwargs)

    def get_greediness(self):
        greediness = self.gene.get_trait(Trait.GREEDINESS)
        if self.signal:
            return max(0, greediness - self.chalones)
        return greediness

    def signaling(self):
        if self.get_cell().chalones > 0:
            self.chalones += 1
            self.get_cell().chalones -= 1

    def get_descendant(self):
        amplification = self.gene.get_trait(Trait.AMPLIFICATION) / GENE_CAPACITY
        if self.chalones > 0:
            amplification /= self.chalones
            self.chalones = 0

        if random_state.uniform(0, 1) <= amplification:
            return Stem(gene=copy.deepcopy(self.gene), dish=self.dish, signal=self.signal)
        else:
            return Terminal(gene=copy.deepcopy(self.gene), dish=self.dish, signal=self.signal)


class Terminal(Cellulata):
    def __init__(self, gene, signal=True, **kwargs):
        self.chromo, self.radius = 0.80, 10  # margenta
        self.signal = signal

        self.radius = 0.008

        super().__init__(gene=gene, **kwargs)

    def signaling(self):
        if self.signal and self.get_time() % self.gene.get_trait(Trait.SIGNALING) == 0:
            self.energy_release += 1
            self.get_cell().chalones += 1

        # no signal
        yield None

    def get_descendant(self):
        return None
