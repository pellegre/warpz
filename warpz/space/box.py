import itertools
from enum import IntEnum
import sys

from warpz.kernel.universe import *
from warpz.kernel.core import *

import numpy
import math
import types

EPSILON = 10e-14


class Direction:
    """"
    unit vector in 3D space

    attributes
    ----------
    magnitude : float
        vector magnitude, since it's an unit vector, it should be 1.0 (or 0.0 for the null vector)
    vector : np.array
        3D unit vector

    """

    def __init__(self, vector, normalize=True):
        """
        parameters
        ----------
        vector : np.array
            each component of the 3D direction, no need to normalize the values to an unit vector
        """

        if normalize:
            magnitude = numpy.linalg.norm(vector)
            if magnitude < EPSILON or magnitude == math.inf:
                self.magnitude = 0
                self.vector = numpy.zeros(numpy.shape(vector))
            else:
                self.vector = vector / magnitude
                self.magnitude = magnitude
        else:
            # careful when using this !
            self.vector = vector
            self.magnitude = numpy.linalg.norm(vector)

    def __str__(self):
        return str(self.vector) + " (" + str(self.magnitude) + ")"


class Particle(Agent):
    CELL = "_cell"
    UNIVERSE = "_universe"

    def __init__(self, **kwargs):
        # base agent
        super().__init__(**kwargs)

        # stacked flag
        self._stacked = False

    def get_universe(self):
        """
        get agent's universe

        """

        return self._universe

    def get_cell(self):
        """
        get agent's cell

        """

        return self._universe.get_cell(self.get_position())

    def is_stacked(self):
        """
        is particle on the stack

        """

        return self._stacked

    def set_stacked(self, stacked):
        self._stacked = stacked

    def interact(self, signals):
        """
        particle interaction with the environment

        """

        yield

    def __call__(self, signals):
        if self.get_time() == 0:
            setattr(self, Agent.TIME, self.get_universe().get_time())

        if not self.is_stacked():
            interaction = self.interact(signals)
            if isinstance(interaction, types.GeneratorType):
                yield from interaction
            elif isinstance(interaction, Iterable):
                yield from interaction


class Ball(Agent):
    CELL = "_cell"
    UNIVERSE = "_universe"

    def __init__(self, radius, **kwargs):
        # base agent
        super().__init__(**kwargs)

        # ball radius
        self.radius = radius

        # stacked flag
        self._stacked = False

        # collisions
        self._collisions = 0

    def get_universe(self):
        """
        get agent's universe

        """

        return self._universe

    def get_collisions(self):
        """
        get agent's collisions

        """

        return self._collisions

    def collide(self):
        """
        count agent's collisions

        """

        self._collisions += 1

    def get_cell(self):
        """
        get agent's cell

        """

        return self._universe.get_cell(self.get_position())

    def is_stacked(self):
        """
        is particle on the stack

        """

        return self._stacked

    def set_stacked(self, stacked):
        self._stacked = stacked

    def interact(self, signals):
        """
        particle interaction with the environment

        """

        yield

    def __call__(self, signals):
        # reset collisions
        self._collisions = 0

        # interact
        if not self.is_stacked():
            interaction = self.interact(signals)
            if isinstance(interaction, types.GeneratorType):
                yield from interaction
            elif isinstance(interaction, Iterable):
                yield from interaction


class Cell(Agent):
    NEIGHBORS = "neighbors"

    def __init__(self, index, universe, limit=0, **kwargs):
        super().__init__(leader=True, **kwargs)

        # set index in the universe
        self.index = index
        self.universe = universe

        # agent's limit (after that, they are stacked)
        self._limit = limit

        # stacked agents
        self._stacked_agents = list()

    def get_stack(self, condition=None):
        """
        get stacked agents

        """

        return filter(condition, self._stacked_agents)

    def clear_stack(self):
        """
        clear stacked agents

        """

        self._stacked_agents.clear()

    def get_stack_size(self):
        """
        count of stacked agents

        """

        return len(self._stacked_agents)

    def push(self, agent):
        """
        push agent to the stack

        parameters
        ----------
        agent : Agent
            agent
        """

        self._stacked_agents.append(agent)

    def pop(self):
        """
        pop agent to the stack

        """

        return self._stacked_agents.pop(0)

    @staticmethod
    def interact(signals):
        """
        cell interaction with the environment

        """
        yield

    def __call__(self, signals):
        # stack agents
        if self._limit > 0:
            agents = 0
            for agent in filter(lambda a: self.universe.get_cell(a.get_position()) is self, self.get_children()):
                # pile up the bastards
                if agents >= self._limit and agent.get_time() == 0:
                    yield Push(child=agent)

                agents += 1

            rest = self._limit - agents
            if rest > 0 and self.get_stack_size() > 0:
                yield Pop(amount=rest)

        interaction = self.interact(signals)
        if isinstance(interaction, types.GeneratorType):
            yield from interaction
        elif isinstance(interaction, Iterable):
            yield from interaction


# box agent
class Box(Universe):
    """"
    Box agent in space

    methods
    -------
    live(signals)
        live signal for the agent
    """

    class Boundary(IntEnum):
        CLOSED = 0
        OPEN = 1

    class Tracking(IntEnum):
        ONE = 1
        QUAD = 2
        NINE = 9

    def __init__(self, dimensions=2, grid=25, cell=Cell, tracking_type=Tracking.QUAD, boundary=Boundary.OPEN,
                 upper_bounds=None, bottom_bounds=None, max_collisions=12, limit=math.inf, **kwargs):
        # init base class
        super().__init__(**kwargs)

        # max amount of collisions after stacking an agent
        self._max_collisions = max_collisions

        # boundary type
        self.boundary = boundary

        # tracking type
        self.tracking_type = tracking_type

        # cell type
        self._cell = cell

        # dimensions
        self._dimensions = dimensions

        # universe bounds
        self._upper_bounds = numpy.ones(self._dimensions) if upper_bounds is None else numpy.array(upper_bounds)
        self._bottom_bounds = numpy.zeros(self._dimensions) if bottom_bounds is None else numpy.array(bottom_bounds)

        # set position
        self._set_box_position()

        # grid size
        if isinstance(grid, tuple):
            self._grid = numpy.array(list(grid))
        else:
            self._grid = numpy.array([grid, grid])

        # agent's limit
        self._limit = limit + numpy.product(self._grid)

        # cell's grid
        self._cells = None
        self._initialize_grid()

    def get_agent_limit(self):
        """
        agent (amount) limit

        """
        return self._limit

    def get_grid(self):
        """
        get box grid

        """
        return self._grid

    def get_dimensions(self):
        """
        get dimensions of the box

        """

        return self._dimensions

    def get_upper_bounds(self):
        """
        get universe upper bounds

        return
        ----------
        upper_bounds : np.array(3, float)
            universe upper bounds
        """

        return self._upper_bounds

    def get_bottom_bounds(self):
        """
        get universe upper bounds

        return
        ----------
        upper_bounds : np.array(3, float)
            universe upper bounds
        """

        return self._bottom_bounds

    def get_cell(self, position):
        """
        get cell on a given position

        parameters
        ----------
        position : position
            position
        """

        # grid index
        relative = (position - self._bottom_bounds) / (self._upper_bounds - self._bottom_bounds)
        index = self._grid * relative

        # agent's cell
        return self._cells[tuple(index.astype(int))]

    def get_cells(self):
        """
        get cells grid

        """

        # agent's cell
        return [cell for cell in self._cells.flatten()]

    def get_neighbors(self, cell):
        """
        get cell neighbors

        parameters
        ----------
        cell : Cell
            cell

        returns
        ----------
        Iterable : cells
            neighbors cells
        """
        return self.get_closed_neighbors(cell)

    def get_tracker_neighbors(self, agent):
        """
        get cell neighbors, which should track an agent

        parameters
        ----------
        agent : Agent
            agent

        returns
        ----------
        Iterable : cells
            neighbors cells
        """

        if self.tracking_type == Box.Tracking.NINE:
            return self.get_closed_neighbors(self.get_cell(agent.get_position()))

        elif self.tracking_type == Box.Tracking.QUAD:
            return self._get_quadrant_neighbors(agent)

        return list()

    def get_closed_neighbors(self, cell):
        """
        get (closed) cell neighbors, no periodic boundaries

        parameters
        ----------
        cell : Cell
            cell

        returns
        ----------
        Iterable : cells
            neighbors cells
        """

        index = cell.index

        neighbors = [((index[0] - 1), (index[1] - 1)),
                     ((index[0] - 1), (index[1] + 1)),
                     ((index[0] + 1), (index[1] - 1)),
                     ((index[0] + 1), (index[1] + 1)),
                     ((index[0] + 0), (index[1] - 1)),
                     ((index[0] + 0), (index[1] + 1)),
                     ((index[0] - 1), (index[1] + 0)),
                     ((index[0] + 1), (index[1] + 0))]

        return [self._cells[i] for i in filter(lambda idx: (0 <= idx[0] < self._grid[0]) and
                                                           (0 <= idx[1] < self._grid[1]), neighbors)]

    def get_periodic_neighbors(self, cell):
        """
        get periodic cell neighbors

        parameters
        ----------
        cell : Cell
            cell

        returns
        ----------
        Iterable : cells
            neighbors cells
        """

        index = cell.index

        neighbors = [((index[0] - 1) % self._grid[0], (index[1] - 1) % self._grid[1]),
                     ((index[0] - 1) % self._grid[0], (index[1] + 1) % self._grid[1]),
                     ((index[0] + 1) % self._grid[0], (index[1] - 1) % self._grid[1]),
                     ((index[0] + 1) % self._grid[0], (index[1] + 1) % self._grid[1]),
                     ((index[0] + 0) % self._grid[0], (index[1] - 1) % self._grid[1]),
                     ((index[0] + 0) % self._grid[0], (index[1] + 1) % self._grid[1]),
                     ((index[0] - 1) % self._grid[0], (index[1] + 0) % self._grid[1]),
                     ((index[0] + 1) % self._grid[0], (index[1] + 0) % self._grid[1])]

        return [self._cells[i] for i in neighbors]

    def transport(self, agent, distance, direction):
        """
        transport agent to a new position

        parameters
        ----------
        agent : Agent
            agent
        distance : float
            transport distance
        direction : array
            direction
        """

        # current cell and parents
        current = self.get_cell(agent.get_position())
        old_parents = set(self.get_tracker_neighbors(agent) + [current])

        # transport agent
        position = agent.get_position() + distance * direction
        if isinstance(agent, Particle):
            self._move_particle(agent, position)

        elif isinstance(agent, Ball):
            # get initial position and move the agent
            initial_position = agent.get_position().copy()
            self._move_particle(agent, position)

            # collide
            self._collide_ball(agent, initial_position)

        # new cell
        cell = self.get_cell(agent.get_position())

        # new parents
        new_parents = set(self.get_tracker_neighbors(agent) + [cell])

        # transport linkage
        if len(new_parents.intersection(old_parents)) != len(old_parents):
            # link new parents
            for new in new_parents:
                new.link(agent)

            # unlink from old parents
            for old in old_parents.difference(new_parents):
                old.unlink(agent)

            if current not in new_parents:
                current.unlink(agent)

            # update cell
            setattr(agent, Particle.CELL, cell)

    def put(self, child):
        """ place agent within this universe and link it to the initial cell

        returns
        -------
        child : Agent
            linked child
        """

        if isinstance(child, Particle):
            # set universe and cell
            setattr(child, Particle.UNIVERSE, self)

            # initialize cells
            self._move_particle(child, child.get_position())

            # get cell
            cell = self.get_cell(child.get_position())

            # link agent and cell
            setattr(child, Particle.CELL, cell)
            self.get_environment().link(cell, child)

            # link with neighbors
            for each in self.get_tracker_neighbors(child):
                self.get_environment().link(each, child)

        elif isinstance(child, Ball):
            # check size
            delta = min((self.get_upper_bounds() - self.get_bottom_bounds()) / self.get_grid())
            if child.radius >= delta:
                raise RuntimeError("can't have ball bigger than grid size - radius " +
                                   str(child.radius) + " - delta " + str(delta))

            # set universe and cell
            setattr(child, Ball.UNIVERSE, self)

            # initialize cells
            initial_position = child.get_position().copy()
            self._move_particle(child, child.get_position())

            # collide ball
            self._collide_ball(child, initial_position)

            # get cell
            cell = self.get_cell(child.get_position())

            # link agent and cell
            setattr(child, Ball.CELL, cell)
            self.get_environment().link(cell, child)

            # link with neighbors
            for each in self.get_tracker_neighbors(child):
                self.get_environment().link(each, child)
        else:
            # it's in the universe
            self.get_environment().link(self, child)

        return child

    @staticmethod
    def get_mesh():
        return None

    def make_cell(self, **kwargs):
        return self._cell(**kwargs)

    def _get_quadrant_neighbors(self, agent):
        cell = self.get_cell(agent.get_position())
        axis_distances = agent.get_position() - cell.get_position()

        indices = [1 if dist >= 0 else -1 for dist in axis_distances]

        neighbors = [tuple([cell.index[i] + indices[i] for i in range(len(indices))]),
                     tuple([cell.index[i] + indices[i] if i == 0 else cell.index[i] for i in range(len(indices))]),
                     tuple([cell.index[i] + indices[i] if i == 1 else cell.index[i] for i in range(len(indices))])]

        return [self._cells[i] for i in filter(lambda idx: (0 <= idx[0] < self._grid[0]) and
                                                           (0 <= idx[1] < self._grid[1]), neighbors)]

    def _move_particle(self, agent, position):
        # test boundaries
        out_of_box_bottom = (position < self._bottom_bounds)
        out_of_box_upper = (position >= self._upper_bounds)

        # check for flat dimensions
        flat_dimensions = numpy.where((self._upper_bounds - self._bottom_bounds) <= 0)

        # cancel out flat dimensions
        out_of_box_bottom[flat_dimensions] = False
        out_of_box_upper[flat_dimensions] = False

        # check if any agent got outside of the universe
        out_bottom, out_upper = numpy.any(out_of_box_bottom), numpy.any(out_of_box_upper)
        if out_bottom or out_upper:
            if out_bottom:
                if self.boundary is Box.Boundary.OPEN:
                    position += out_of_box_bottom * (self._upper_bounds - self._bottom_bounds)
                else:
                    position = numpy.array([max(self._bottom_bounds[i] + 0.001, position[i])
                                            for i in range(0, len(position))])
            if out_upper:
                if self.boundary is Box.Boundary.OPEN:
                    position -= out_of_box_upper * (self._upper_bounds - self._bottom_bounds)
                else:
                    position = numpy.array([min(self._upper_bounds[i] - 0.001, position[i])
                                            for i in range(0, len(position))])

            # set agent's position
            setattr(agent, Agent.POSITION, position)
            if self.boundary is Box.Boundary.OPEN:
                self._move_particle(agent, position)
        else:
            # set agent's position
            setattr(agent, Agent.POSITION, position)

    def _collide_ball(self, agent, initial_position):
        sys.setrecursionlimit(15000)

        # check maximum amount of collisions
        if agent.get_collisions() > self._max_collisions:
            return

        # current movement
        movement = numpy.linalg.norm(initial_position - agent.get_position())

        if self.tracking_type is Box.Tracking.ONE:
            # get ball extremes
            radius, position = agent.radius, agent.get_position()
            corners = [(position[0] - radius, position[1]), (position[0] + radius, position[1]),
                       (position[0], position[1] - radius), (position[0], position[1] + radius),
                       (position[0] - radius, position[1] + radius), (position[0] + radius, position[1] + radius),
                       (position[0] - radius, position[1] - radius), (position[0] + radius, position[1] - radius)]

            # cells
            cells = {self.get_cell(c) for c in corners if (self._bottom_bounds[0] < c[0] < self._upper_bounds[0]) and
                     (self._bottom_bounds[1] < c[1] < self._upper_bounds[1])}

            # add current one
            cells.add(self.get_cell(agent.get_position()))
        else:
            # add all parent cells
            cells = set()
            try:
                cells.update(agent.get_parents())
            except OSError:
                cells.add(self.get_cell(agent.get_position()))

        for cell in cells:
            for other in filter(lambda a: a is not agent,
                                cell.get_agents(condition=lambda a: isinstance(a, Ball))):
                # check maximum amount of collisions
                if other.get_collisions() < self._max_collisions and agent.get_collisions() < self._max_collisions:
                    # get positions
                    one_position, other_position = agent.get_position(), other.get_position()

                    # get collision direction
                    direction = one_position - other_position
                    norm = numpy.linalg.norm(direction)

                    # combined diameter
                    diameter = 2 * (other.radius + agent.radius)
                    if norm < diameter:
                        # calculate distances
                        one_distance = max((other.radius**2 / agent.radius**2) * movement / 2, diameter / 3)
                        other_distance = max((agent.radius**2 / other.radius**2) * movement / 2, diameter / 3)

                        # normalize direction
                        if norm > EPSILON:
                            direction /= numpy.linalg.norm(direction)
                        else:
                            theta = numpy.random.uniform(0, 2 * math.pi)
                            direction = numpy.array([math.cos(theta), math.sin(theta)])

                        # count collisions
                        other.collide()
                        agent.collide()

                        # trigger transport of the collided ball
                        self.transport(other, other_distance / math.sqrt(other.get_collisions()), -direction)
                        self.transport(agent, one_distance / math.sqrt(agent.get_collisions()), direction)

    def _initialize_grid(self):
        # initialize cell grid
        if self._cells is None:
            self._cells = numpy.empty(tuple(self._grid), object)

            # cell index
            index = [list(range(0, self._grid[0]))]
            for n in range(1, self._dimensions):
                index.append(list(range(0, self._grid[n])))

            # put it
            for idx in itertools.product(*index):
                stride = (self._upper_bounds - self._bottom_bounds) / self._grid
                position = stride / 2 + numpy.array(idx) * stride + self._bottom_bounds

                # create cell
                cell = self.make_cell(index=idx, universe=self, position=position)

                # link cell
                self._cells[idx] = self.get_environment().link(self, cell)

            # define neighbors
            for idx in itertools.product(*index):
                setattr(self._cells[idx], Cell.NEIGHBORS, self.get_periodic_neighbors(self._cells[idx]))

    def _set_box_position(self):
        # calculate position as an average between the bounds
        self._position = (self._upper_bounds + self._bottom_bounds) / 2


class Transport(Kernel):
    def __init__(self, distance, direction):
        # transport vector
        self.vector = Direction(direction).vector
        self.distance = distance
        super().__init__()

    def __call__(self, agent):
        if isinstance(agent, Particle):
            # get universe and cell
            universe = getattr(agent, Particle.UNIVERSE, None)

            # transport particle
            universe.transport(agent, self.distance, self.vector)

        elif isinstance(agent, Ball):
            # get universe and cell
            universe = getattr(agent, Ball.UNIVERSE, None)

            # transport ball
            universe.transport(agent, self.distance, self.vector)

        else:
            raise RuntimeError("can't transport agent " + type(agent).__name__)


class Replicate(Kernel):
    def __init__(self, born, **kwargs):
        # transport vector
        self.born = born
        super().__init__(**kwargs)

    def __call__(self, agent):
        cell = agent.get_cell()
        if isinstance(agent, Particle):
            # get universe and cell
            universe = getattr(agent, Particle.UNIVERSE, None)

            # check hard limit
            if universe.get_count() < universe.get_agent_limit():
                # replicate context
                setattr(self.born, Agent.POSITION, agent.get_position())
                setattr(self.born, Particle.UNIVERSE, universe)
                setattr(self.born, Particle.CELL, cell)

                # place new born on the universe
                cell.link(self.born)

                # link with neighbors
                for each in universe.get_tracker_neighbors(self.born):
                    each.link(self.born)

        elif isinstance(agent, Ball):
            # get universe and cell
            universe = getattr(agent, Ball.UNIVERSE, None)

            # check hard limit
            if universe.get_count() < universe.get_agent_limit():
                # replicate context
                setattr(self.born, Agent.POSITION, agent.get_position())
                setattr(self.born, Ball.UNIVERSE, universe)
                setattr(self.born, Ball.CELL, cell)

                # place new born on the universe
                cell.link(self.born)

                # link with neighbors
                for each in universe.get_tracker_neighbors(self.born):
                    each.link(self.born)

        else:
            raise RuntimeError("can't replicate agent " + type(agent).__name__)


class Push(Kernel):
    def __init__(self, child):
        # child
        self.child = child
        super().__init__()

    def __call__(self, cell):
        # push child to the stack
        cell.push(self.child)
        self.child.set_stacked(True)

        # remove it from the environment
        self.child.remove()


class Pop(Kernel):
    def __init__(self, amount):
        # pop at least this amount
        self.amount = amount
        super().__init__()

    def __call__(self, cell):
        if cell.get_stack_size() > 0:
            # popping bottles with modelsz
            amount = min(cell.get_stack_size(), self.amount)
            for i in range(0, amount):
                # pop it and link
                child = cell.pop()
                child.set_stacked(False)

                # link child
                cell.link(child)

                # link with neighbors
                for each in cell.universe.get_tracker_neighbors(child):
                    each.link(child)
