from warpz.space.box import *
from enum import IntEnum
from bisect import bisect

from models.migrate.genes import *

import numpy
import time

# random state
random_state = numpy.random.RandomState()

# motility
MOTILITY = 0.00007

# cell radius
RADIUS = 0.0006

# cellulata grace period
GRACE_PERIOD = 100

# seed pool
seed_pool = GenePool(genes=seed_gene_pool, capacity=GENE_CAPACITY)

# ===========
#
# Material unit
#
# ===========

CHEMOKINE = "chemokine"
GLUCOSE = "glucose"


class Unit(Cell):
    def __init__(self, **kwargs):
        # material stack for this unit
        self.material = {CHEMOKINE: 0, GLUCOSE: 0}
        super().__init__(**kwargs)

    def interact(self, signals):
        # physical assertion
        for chemokine in self.material:
            if self.material[chemokine] > 0 and chemokine != GLUCOSE:
                self.material[chemokine] -= min(self.universe.cost, self.material[chemokine])

            assert self.material[chemokine] >= 0

    def get_glucose(self, amount):
        # get glucose stack
        glucose_stack = self.material[GLUCOSE]
        if glucose_stack >= amount:
            # give it as it is
            self.material[GLUCOSE] -= amount
            return amount

        else:
            # give what it is
            self.material[GLUCOSE] = 0
            return glucose_stack


# ===========
#
# Medium (collection of material units)
#
# ===========

class Medium(Box):
    def __init__(self, glucose, grid=10, diffusion_rate=0.01, diffusion_period=20, gene_pool=seed_pool, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Unit, **kwargs)

        # gene pool
        self.gene_pool = gene_pool

        # total of glucose
        self.glucose = glucose

        # uniformly distribute glucose
        self.uniform_distribution(filled=False)

        # diffusion parameters
        self.diffusion_rate, self.diffusion_period = diffusion_rate, diffusion_period

        # food and energy release
        self.mesh = numpy.zeros(self._dimensions * (self._grid,), float)

        # initial time
        self.tic, self.toc = time.process_time(), time.process_time()

        # cost function
        self.clock_time, self.cost, self.accumulated_cost = 0, 1, 0

    def put(self, child):
        # put child on universe
        super().put(child)

        # take glucose
        child.get_cell().material[GLUCOSE] -= child.glucose

        # adjust glucose
        remain = child.get_cell().material[GLUCOSE]
        if remain < 0:
            child.glucose += remain
            # fix up level
            child.get_cell().material[GLUCOSE] = 0

    def uniform_distribution(self, filled=True):
        # get cells
        if filled:
            children = self.get_children(condition=lambda c: len(c.get_children()) > 0)
        else:
            children = self.get_children()

        # distribute glucose
        if len(children):
            each = self.glucose // len(children)

            # distribute energy
            for cell in self.get_children():
                amount = min(each, self.glucose)
                self.glucose -= amount
                cell.material[GLUCOSE] += amount

        # balance checking
        assert self.glucose >= 0

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
            self.mesh[each.index[1], each.index[0]] = math.log(each.material[CHEMOKINE] + 1)

        # get light mesh
        return self.mesh

    def __call__(self, signals):
        # diffuse
        if self.get_time() % self.diffusion_period == 0:
            self.diffuse()

        # update simulation cost
        self.update_cost()

        # periodic distribution of glucose
        if self.get_time() % 15 == 0:
            self.uniform_distribution()


# ===========
#
# Control unit
#
# ===========

class Action(IntEnum):
    MOVE = 0
    IDLE = 1


class ControlUnit:
    @staticmethod
    def get_angle(direction):
        # get coordinates from direction
        x, y = direction[0], direction[1]

        if y >= 0:
            # first and second quadrant
            return math.acos(x)
        else:
            return 2 * math.pi - math.acos(x)

    def __init__(self, gene, theta, q_matrix=None, **kwargs):
        # gene
        self.gene = copy.deepcopy(gene)

        # mutate machine
        mutation = self.gene.get_trait(Trait.MUTATION) / GENE_CAPACITY
        if random_state.uniform(0, 1) < mutation:
            self.gene.mutate(random_state)

        # propagate state and decision units information
        self.gene.set_trait(Trait.STATES, gene.get_trait(Trait.STATES))
        self.gene.set_trait(Trait.UNITS, gene.get_trait(Trait.UNITS))

        # states size
        self.states_size = self.gene.get_trait(Trait.STATES)

        # state distribution
        self.phi_state = numpy.linspace(0, 2.0 * math.pi, self.states_size + 1)

        # action taken
        self.action_taken, self.state = None, None

        # player direction (theta angle from x-axis)
        self.direction = numpy.array([math.cos(theta), math.sin(theta)])
        self.theta = theta

        # Q learning matrix
        if q_matrix is None:
            self.q_matrix = numpy.random.uniform(0.00, 0.10, (self.states_size, 2))
        else:
            self.q_matrix = numpy.copy(q_matrix)

        # epsilon (switching from greedy to random behavior)
        self.epsilon = self.gene.get_trait(Trait.EPSILON) / GENE_CAPACITY

        # Q parameters
        self.gamma = self.gene.get_trait(Trait.GAMMA) / GENE_CAPACITY
        self.alpha = self.gene.get_trait(Trait.ALPHA) / GENE_CAPACITY

        # agent instantiation
        super().__init__(**kwargs)

    def get_state(self, gradient):
        norm = numpy.linalg.norm(gradient)
        if norm > 0:
            # direction
            direction = gradient / norm
            phi = self.get_angle(direction / numpy.linalg.norm(direction))

            # phi (discrete) state
            state = min(bisect(self.phi_state, phi), self.states_size - 1)

            # calculate state
            return state

        else:
            # random state
            return numpy.random.randint(0, self.states_size)

    def update_state(self, gradient, reward=0.0):
        # get state from gradient
        state = self.get_state(gradient)

        # update Q matrix
        if self.state is not None:
            a = int(self.action_taken)
            self.q_matrix[self.state, a] = \
                self.q_matrix[self.state, a] + self.alpha * \
                (reward + self.gamma * numpy.max(self.q_matrix[state, :]) - self.q_matrix[self.state, a])

        # keep track of states
        self.state = state

    def action(self):
        # sample and press button
        if random_state.uniform(0, 1) < self.epsilon:
            # random move
            if random_state.uniform(0, 1) >= 0.50:
                # press button
                self.action_taken = Action.MOVE
            else:
                # idle
                self.action_taken = Action.IDLE

        else:
            # choose action by maximizing utility
            self.action_taken = max({Action.MOVE, Action.IDLE},
                                    key=lambda action: self.q_matrix[self.state, int(action)])

        # get action
        return self.action_taken


# ===========
#
# Cellulata
#
# ===========

class CellulaTally:
    def __init__(self):
        # tallied variables
        self.tallies, self.values, self.counter = dict(), dict(), dict()

    def add(self, tallies):
        for tally in tallies:
            # update tallies
            if tally in self.tallies:
                self.counter[tally] += 1
                self.tallies[tally] += tallies[tally]
                self.values[tally] = tallies[tally]
            else:
                self.counter[tally] = 1
                self.tallies[tally] = tallies[tally]
                self.values[tally] = tallies[tally]

    def average(self):
        return {tally: self.tallies[tally] / self.counter[tally] for tally in self.tallies}

    def total(self):
        return {tally: self.tallies[tally] for tally in self.tallies}

    def value(self):
        return {tally: self.values[tally] for tally in self.tallies}


class Cellulata(Ball):
    # cellulata's color
    RELEASING_COLOR = 0.20
    GREEDY_COLOR = 0.80

    @staticmethod
    def is_null_vector(vector):
        return math.fabs(vector[0]) < EPSILON and math.fabs(vector[1]) < EPSILON

    @staticmethod
    def build(gene, **kwargs):
        return Cellulata(gene=gene, **kwargs)

    @staticmethod
    def build_greedy(gene, **kwargs):
        return Cellulata(gene=gene, chemokines=0, **kwargs)

    @staticmethod
    def build_releasing(gene, **kwargs):
        # build it with the bipolar threshold
        bipolar_threshold = BIPOLARITY_FACTOR * gene.get_trait(Trait.BIPOLARITY) * gene.get_trait(Trait.CONSUMPTION)
        return Cellulata(gene=gene, chemokines=bipolar_threshold, **kwargs)

    def __init__(self, gene, chemokines=0, glucose=None, units=None, replication=True, releasing_mode=False, **kwargs):
        # cellulata gene
        self.gene = gene

        # chemokines stack
        self.chemokines = chemokines

        # cellulata motility
        self.motility = MOTILITY * self.gene.get_trait(Trait.MOTILITY)

        # releasing state
        if self.hit_releasing() or releasing_mode:
            # red
            self.color = Cellulata.RELEASING_COLOR

            # release chemokine (releasing mode)
            self.releasing_mode = True
        else:
            # pink
            self.color = Cellulata.GREEDY_COLOR

            # consume chemokine (greedy mode)
            self.releasing_mode = False

        # direction angle
        self.direction, self.previous_direction = numpy.zeros(2), numpy.zeros(2)

        # mutation rate
        self.mutation_rate = self.gene.get_trait(Trait.MUTATION) / GENE_CAPACITY
        # crossover rate
        self.crossover_rate = self.gene.get_trait(Trait.CROSSOVER) / GENE_CAPACITY

        # total control units
        if units is None:
            # new control unit instance
            self.total_units = self.gene.get_trait(Trait.UNITS)
            self.units = [ControlUnit(theta=theta, gene=self.gene)
                          for theta in numpy.linspace(0, 2 * math.pi, self.total_units, endpoint=False)]
        else:
            # propagate control unit state
            self.total_units = len(units)
            self.units = [ControlUnit(theta=unit.theta, gene=self.gene, q_matrix=unit.q_matrix) for unit in units]

        # state of the cell
        self.gradient, self.reward = None, 0.0

        # glucose consumption
        if glucose is None:
            self.glucose = self.gene.get_trait(Trait.BORN)
        else:
            self.glucose = glucose

        # born time
        self.born_time = None

        # apoptosis flag (surgery)
        self.apoptosis = False

        # replication flag
        self.replication = replication

        # tallied variables
        self.tallies = CellulaTally()

        # agent instantiation
        radius = RADIUS + 3 * RADIUS * self.gene.get_trait(Trait.RADIUS) / GENE_CAPACITY

        # born gene
        self.born_gene = copy.deepcopy(self.gene)

        # initialize agent
        super().__init__(radius=radius, **kwargs)

        # initialize previous position
        self.initial_position, self.previous_position, self.collisions = self.get_position(), self.get_position(), 1

    def is_greedy(self):
        return not self.releasing_mode

    def is_releasing(self):
        return self.releasing_mode

    def hit_releasing(self):
        # releasing condition
        hit_releasing = \
            (self.chemokines >= BIPOLARITY_FACTOR *
             self.gene.get_trait(Trait.BIPOLARITY) * self.gene.get_trait(Trait.CONSUMPTION))
        if hit_releasing:
            self.color = Cellulata.RELEASING_COLOR

        # return state
        return hit_releasing

    def hit_greedy(self):
        # releasing condition
        hit_greedy = (self.chemokines == 0)
        if hit_greedy:
            self.color = Cellulata.GREEDY_COLOR

        # return state
        return hit_greedy

    def resolve_gradient(self, cell, current):
        if self.get_universe().boundary is Box.Boundary.CLOSED:
            # direct gradient
            gradient = cell.get_position() - self.get_position()
        else:
            # take into account periodic conditions
            if abs(current.index[0] - cell.index[0]) > 1 or abs(current.index[1] - cell.index[1]) > 1:
                # reverse the gradient, thanks chad !
                gradient = self.get_position() - cell.get_position()
            else:
                # direct gradient
                gradient = cell.get_position() - self.get_position()

        return gradient

    def get_gradient(self):
        # get current cell
        cell = self.get_cell()

        # get neighboring cells
        if self.get_universe().boundary is Box.Boundary.CLOSED:
            cells = self.get_universe().get_closed_neighbors(cell)
        else:
            cells = self.get_universe().get_periodic_neighbors(cell)

        # follow the gradient
        idx = numpy.argmax([each.material[CHEMOKINE] for each in cells])

        # gradient
        gradient = self.resolve_gradient(cell=cells[idx], current=cell)

        # normalize gradient
        norm = numpy.linalg.norm(gradient)
        if norm > EPSILON:
            gradient /= norm

        # set up gradient
        return gradient

    def transport(self, units=1):
        # reset direction
        self.direction = numpy.zeros(2)

        # control units
        for each in self.units:
            action = each.action()
            if action is Action.MOVE:
                # accumulate direction
                self.direction += each.direction

        # get reward based on gradient sensing
        gradient_norm = numpy.linalg.norm(self.gradient)

        # fix angle
        direction_norm = numpy.linalg.norm(self.direction)
        if direction_norm > EPSILON:
            # transport
            self.direction /= direction_norm

        # tune up motility
        yield Transport(units * self.motility, self.direction)

    def gone(self):
        # put back remaining glucose
        if self.glucose > 0:
            self.get_cell().material[GLUCOSE] += self.glucose

        # release chemokine only when there is not apoptosis
        if not self.apoptosis:
            self.get_cell().material[CHEMOKINE] += self.chemokines

        # cellula is done
        yield Gone()

    def sense(self):
        # set reward
        movement = self.get_position() - self.previous_position

        # normalize direction
        movement_norm = numpy.linalg.norm(movement)
        if movement_norm > 0:
            movement /= movement_norm

        # get reward
        self.reward = numpy.dot(movement, self.gradient) if self.gradient is not None else 0

        # sense gradient
        self.gradient = self.get_gradient()

        # update control units reward
        for each in self.units:
            each.update_state(self.gradient, self.reward)

    def done(self):
        # setup born time
        current = self.get_universe().get_time()
        if self.born_time is None:
            self.born_time = current

        # grace period on start
        return self.apoptosis or self.glucose == 0

    def release(self):
        # release chemokines
        if self.releasing_mode:
            # release it
            released = min(self.chemokines, self.gene.get_trait(Trait.RELEASE))
            self.get_cell().material[CHEMOKINE] += released

            # balance internal supply
            self.chemokines -= released

            if self.hit_greedy():
                # bipolar switch
                self.releasing_mode = False

    def consume(self):
        # get current cell
        cell = self.get_cell()

        # consume chemokines
        if not self.releasing_mode:
            # consume chemokines
            consumed = min(cell.material[CHEMOKINE], self.gene.get_trait(Trait.CONSUMPTION))

            # balance cell
            cell.material[CHEMOKINE] -= consumed

            # produce chemokines
            produced = self.gene.get_trait(Trait.PRODUCTION)

            # balance chemokines
            self.chemokines += produced + consumed

            if self.hit_releasing():
                # bipolar switch
                self.releasing_mode = True

        # request glucose
        glucose = self.get_universe().cost + self.gene.get_trait(Trait.GREEDINESS)
        self.glucose += cell.get_glucose(glucose)

    def metabolize(self):
        if self.glucose > 0:
            # metabolize glucose
            metabolized = min(self.glucose, self.get_universe().cost)

            # balance glucose
            self.glucose -= metabolized
            self.get_universe().glucose += metabolized

    def mate(self):
        # sample crossover
        if numpy.random.uniform(0, 1) < self.crossover_rate:
            # get current cell
            cell = self.get_cell()

            # and neighbors
            neighbors = cell.get_children(condition=lambda c: isinstance(c, Cellulata))

            # randomly sample one
            if len(neighbors) > 0:
                other = neighbors[numpy.random.randint(0, len(neighbors))]

                # sample crossover
                if numpy.random.uniform(0, 1) < other.crossover_rate:
                    # perform crossover
                    self.born_gene.crossover(other.gene, random_state)

                    # tally crossover
                    self.tallies.add({"crossover-idv": 1})

    def replicate(self):
        # release the child when enough chemokines has been harvested
        if self.replication and self.glucose >= 2 * self.gene.get_trait(Trait.BORN):
            # balance glucose
            born_glucose = self.glucose // 2
            self.glucose -= born_glucose

            if random_state.uniform(0, 1) < self.mutation_rate:
                # create random gene
                gene = self.get_universe().gene_pool.get_random_gene(random_state)

                # tally random individual
                self.tallies.add({"mutation-idv": 1})

                # replicate (random descendant)
                yield Replicate(born=self.build(chemokines=self.chemokines // 2, gene=gene, glucose=born_glucose,
                                                position=self.get_position(), releasing_mode=self.releasing_mode))

            else:
                # mate
                self.mate()

                # mutate gene
                if random_state.uniform(0, 1) < self.mutation_rate:
                    self.born_gene.mutate(random_state)

                    # tally random trait
                    self.tallies.add({"mutation-trt": 1})

                # propagate state and decision units information
                self.born_gene.set_trait(Trait.STATES, self.gene.get_trait(Trait.STATES))
                self.born_gene.set_trait(Trait.UNITS, self.gene.get_trait(Trait.UNITS))

                # replicate
                yield Replicate(born=self.build(chemokines=self.chemokines // 2, gene=self.born_gene,
                                                glucose=born_glucose, units=self.units, position=self.get_position(),
                                                releasing_mode=self.releasing_mode))

            # reset chemokine
            self.chemokines //= 2

            # reset crossover state
            self.born_gene = copy.deepcopy(self.gene)

        # verify physical sanity
        assert self.glucose >= 0

    def tally(self):
        # calculate cosine with previous direction
        if not self.is_null_vector(self.previous_direction) and not self.is_null_vector(self.direction):
            # get cosine
            cosine = numpy.dot(self.direction, self.previous_direction)

            # get distance from previous position
            centroid_change = self.get_position() - self.previous_position

            # get distance (total and current)
            distance = numpy.linalg.norm(centroid_change)

            # projection towards previous direction
            delta_x = numpy.dot(centroid_change, self.previous_direction)

            # y projection
            norm = numpy.linalg.norm(self.previous_direction)
            delta_y = numpy.cross(centroid_change, self.previous_direction) / norm

            # tally persistence
            self.tallies.add({"cosine": cosine, "delta_x": delta_x, "delta_y": delta_y,
                              "distance": distance})

        # record previous state
        self.previous_direction = self.direction
        self.previous_position = self.get_position()

    def interact(self, signals):
        # interaction loop
        if not self.done():
            # sense surroundings
            yield self.sense()

            # release
            self.release()

            # consume
            self.consume()

            # metabolize
            self.metabolize()

            # replicate
            yield self.replicate()

            # tally behavior
            self.tally()

            # move
            yield self.transport()
        else:
            # gone
            yield self.gone()


# ===========
#
# Synchro Game
#
# ===========

class Pointer(Cellulata):
    def __init__(self, pointer=None, units_number=8, states_number=16, motility=None, **kwargs):
        # peer pointer
        self.pointer = pointer

        # get a gene
        gene = seed_pool.get_random_gene(random_state)

        # turn off unused traits
        gene.set_trait(Trait.BORN, 0)
        gene.set_trait(Trait.GREEDINESS, 0)
        gene.set_trait(Trait.MUTATION, 0)
        gene.set_trait(Trait.CROSSOVER, 0)

        # automata genes
        gene.set_trait(Trait.EPSILON, 60)
        gene.set_trait(Trait.ALPHA, 200)
        gene.set_trait(Trait.GAMMA, 200)
        gene.set_trait(Trait.STATES, states_number)
        gene.set_trait(Trait.UNITS, units_number)
        gene.set_trait(Trait.RADIUS, GENE_CAPACITY)

        # consumption and production rate
        gene.set_trait(Trait.CONSUMPTION, 0)
        gene.set_trait(Trait.BIPOLARITY, 0)

        # release
        gene.set_trait(Trait.RELEASE, 30)
        gene.set_trait(Trait.PRODUCTION, 30)

        # set maximum motility
        gene.set_trait(Trait.MOTILITY, 120)

        # initialize cellulata
        super(Pointer, self).__init__(replication=False, gene=gene, **kwargs)

        # setup motility of given
        if motility is not None:
            self.motility = motility

    def hit_releasing(self):
        # return state
        return False

    def hit_greedy(self):
        # return state
        return False

    def done(self):
        # always go on
        return False

    def get_pointer(self):
        return next(iter(self.get_universe().get_agents(
            condition=lambda a: isinstance(a, Cellulata) and a is not self)))

    def get_direction(self):
        raise RuntimeError("get_direction not implemented")

    def get_gradient(self):
        # run from the catcher
        direct_gradient = self.get_direction()

        # normalize gradient
        direct_norm = numpy.linalg.norm(direct_gradient)

        # direct gradient
        if direct_norm > EPSILON:
            direct_gradient /= direct_norm

        # return gradient
        return direct_gradient

    def interact(self, signals):
        # search for pointer
        if self.pointer is None:
            self.pointer = self.get_pointer()

        # chemokines stack
        if self.releasing_mode:
            self.chemokines += 2048

        # cellulata interaction
        yield from super(Pointer, self).interact(signals)


# ===========
#
# Target Pointer
#
# ===========

class Target(Pointer):
    def __init__(self, releasing_mode=False, **kwargs):
        super().__init__(releasing_mode=releasing_mode, **kwargs)
        # set releasing color
        self.color = Cellulata.RELEASING_COLOR

    def get_direction(self):
        return self.get_position() - self.pointer.get_position()


# ===========
#
# Leader Pointer
#
# ===========

class Leader(Pointer):
    def __init__(self, direction, releasing_mode=False, **kwargs):
        super().__init__(releasing_mode=releasing_mode, **kwargs)
        # set releasing color
        self.color = Cellulata.RELEASING_COLOR

        # set fixed direction
        self.movement = direction

    def get_direction(self):
        return self.movement


# ===========
#
# Catcher Pointer
#
# ===========

class Catcher(Pointer):
    def __init__(self, **kwargs):
        super().__init__(releasing_mode=False, **kwargs)
        # set greedy color
        self.color = Cellulata.GREEDY_COLOR

    def get_direction(self):
        return self.pointer.get_position() - self.get_position()
