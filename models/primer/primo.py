from warpz.simulation.local import *
from warpz.space.box import *

import math
import sympy
import numpy
import networkx

from networkx.drawing.nx_agraph import graphviz_layout


random_state = numpy.random.RandomState()


GRID = 50

DISTANCE = GRID / 600
DELTA = DISTANCE / 38

ANGLE = numpy.array([0, 1])


class Interact(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def get_random_direction():
        # random direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def __call__(self, other):
        # get source
        who = self.get_source()

        # get direction
        direction = who.get_position() - other.get_position()
        norm = numpy.linalg.norm(direction)
        if norm < EPSILON:
            direction = self.get_random_direction()

        yield who.move(-direction)


class Push(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, other):
        # get source
        who = self.get_source()
        yield who.move(numpy.array([1, 0]))


class Node(Particle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.branches = {}
        self.attractor = None

    @staticmethod
    def move(direction, distance=DELTA):
        yield Transport(distance=distance, direction=direction)

    def is_reachable(self, node):
        return numpy.linalg.norm(node.get_position() - self.get_position()) <= DISTANCE / 2

    def is_level_reachable(self, node):
        return node.get_position()[0] < self.get_position()[0] or \
               numpy.linalg.norm(node.get_position()[0] - self.get_position()[0]) <= DISTANCE / 2

    def __call__(self, signals):
        if self.attractor is not None:
            direction = self.attractor - self.get_position()
            norm = numpy.linalg.norm(direction)

            if norm > 0:
                yield self.move(direction / norm, distance=DISTANCE/10)

class Prime(Node):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.color = 0.4
        self.radius = 1
        self.value = value


class Connector(Node):
    def __init__(self, **kwargs):
        self.color = 0.9
        self.radius = 0.5
        super().__init__(**kwargs)


# ===========
#
# Screen (with a parcel mesh)
#
# ===========

class Parcel(Cell):
    def __init__(self, **kwargs):
        # initial stack of energy
        self.light, self.initial = 0, 0
        super().__init__(**kwargs)

    def __call__(self, signals):
        pass


class Screen(Box):
    def __init__(self, grid=10, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Parcel, **kwargs)

        # food and energy release
        self.mesh = numpy.zeros(self._dimensions * (self._grid,), float)
        self.root = None
        self.g = networkx.DiGraph()

        self.signal_path = dict()

    def get_mesh(self):
        # mesh
        for each in self.get_children(condition=lambda c: isinstance(c, Parcel)):
            self.mesh[each.index[1], each.index[0]] = each.light

        # get light mesh
        return self.mesh

    def add_node(self, flow, node=None, current=0, power=0):
        if node is None:
            node = self.root

        edge = flow[0]
        value = current + edge * (2 ** power)

        if len(flow) == 1:
            position = node.get_position() + DISTANCE * ANGLE

            prime = Prime(position=position, value=value)
            self.put(prime)
            node.branches[1] = prime

            self.g.add_node(prime.get_id())
            self.g.add_edge(node.get_id(), prime.get_id())

        else:
            if edge in node.branches:
                successor = node.branches[edge]

            else:
                position = node.get_position() + DISTANCE * ANGLE

                successor = Connector(position=position)
                self.put(successor)

            node.branches[edge] = successor

            self.g.add_node(successor.get_id())
            self.g.add_edge(node.get_id(), successor.get_id())

            self.add_node(flow[1:], successor, value, power + 1)

    @staticmethod
    def get_binary_flow(n):
        binary = "{0:032b}".format(n)
        stack = [int(c) for c in binary[binary.find('1'):][::-1]]

        return stack

    def __call__(self, signals):
        prime = sympy.sieve[self.get_time() + 1]
        self.add_node(self.get_binary_flow(prime))

        pos = graphviz_layout(self.g, prog="dot")

        max_width_node, min_width_node = max(pos, key=lambda n: pos[n][0]), min(pos, key=lambda n: pos[n][0])
        max_height_node, min_height_node = max(pos, key=lambda n: pos[n][1]), min(pos, key=lambda n: pos[n][1])

        left, width = pos[min_width_node][0], pos[max_width_node][0]
        bottom, height = pos[min_height_node][1], pos[max_height_node][1]

        if self.get_time() % 10 == 0:
            for node in self.get_agents(condition=lambda n: isinstance(n, Node)):
                if node.get_id() is not None:
                    x, y = pos[node.get_id()]
                    node.attractor = numpy.array([(width - x) / width, (height - y) / height])

# ===========
#
# Simulation
#
# ===========


class Primer(Simulation):
    def __init__(self):
        # main screen
        self.screen = Screen(grid=GRID, boundary=Box.Boundary.CLOSED)

        self.screen.root = Connector(position=numpy.array([0.50, 0.03]))
        self.screen.put(self.screen.root)

        # init base class
        super().__init__(universe=self.screen, agents_filter=lambda agent: isinstance(agent, Node),
                         min_radius=9, max_radius=14)

    def should_draw_vertex(self):
        return True

    def set_vertex_on_agents(self, vertices, agent_index):
        for agent in self.screen.get_agents(condition=lambda a: isinstance(a, Node)):
            if 1 in agent.branches:
                vertices[agent_index[agent], agent_index[agent.branches[1]]] = True
            if 0 in agent.branches:
                vertices[agent_index[agent], agent_index[agent.branches[0]]] = True

    @staticmethod
    def get_random_position():
        return numpy.array([numpy.random.uniform(0, 0.1), numpy.random.uniform(0, 0.1)])

    def should_plot(self):
        return False

    def get_mesh(self):
        return None

    def done(self):
        return False

    def step(self):
        if self.screen.get_time() % 100 == 0:
            # print some stats
            print("[.] ---- time :", self.screen.get_time())
