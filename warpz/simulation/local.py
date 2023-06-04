import numpy
import math

from enum import IntEnum


class Simulation:
    # epsilon
    EPSILON = 10E-8

    # properties
    class Properties(IntEnum):
        CHROMO = 1
        RADIUS = 2
        SCALE_CHROMO = 3

    def __init__(self, universe, min_radius=12, max_radius=16, plotter_period=750, agents_filter=None,
                 width=1390, height=1390, plotter_width=1024, plotter_height=512):
        # plotter width and height
        self.plotter_width = plotter_width
        self.plotter_height = plotter_height

        # windows width and height
        self.width = width
        self.height = height

        # set radius extremes
        self.min_radius = min_radius
        self.max_radius = max_radius

        # plotter period
        self.plotter_period = plotter_period

        # get top node and create environment
        self.universe = universe

        # agent's filter to track positions and observables
        self.agents_filter = agents_filter

        # set default agent's properties
        self.properties = getattr(self, "properties", dict())

        if Simulation.Properties.CHROMO not in self.properties:
            self.properties[Simulation.Properties.CHROMO] = lambda agent: \
                getattr(agent, "color", 0.8)

        if Simulation.Properties.SCALE_CHROMO not in self.properties:
            self.properties[Simulation.Properties.SCALE_CHROMO] = lambda agent: 1.0

        if Simulation.Properties.RADIUS not in self.properties:
            self.properties[Simulation.Properties.RADIUS] = lambda agent: \
                getattr(agent, "radius", 1)

        # update bottom bounds
        self.bottom_bounds = numpy.zeros(3, float)
        self.bottom_bounds[:2] = self.universe.get_bottom_bounds()

        # update upper bounds
        self.upper_bounds = numpy.zeros(3, float)
        self.upper_bounds[:2] = self.universe.get_upper_bounds()

        # update simulation
        self._update_simulation()

    def set_property(self, prop, value):
        if not hasattr(self, "properties"):
            self.properties = dict()

        if not callable(value):
            self.properties[prop] = lambda agent: value
        else:
            self.properties[prop] = value

    @staticmethod
    def should_plot():
        return True

    @staticmethod
    def should_draw_vertex():
        return True

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def get_plotter_width(self):
        return self.plotter_width

    def get_plotter_height(self):
        return self.plotter_height

    def get_upper_bounds(self):
        return self.upper_bounds

    def get_bottom_bounds(self):
        return self.bottom_bounds

    def get_agents_position(self):
        return self._agents_position

    def get_agents_vertex(self):
        return self._agents_vertex

    def set_vertex_on_agents(self, vertices, agent_index):
        pass

    def get_chromo(self):
        return self._visual_chromo

    def get_radius(self):
        return self._visual_radius

    def get_mesh(self):
        return self.universe.get_mesh()

    def get_time(self):
        return self.universe.get_time()

    def run(self):
        # update simulation
        self._update_simulation()

        # run simulation
        self.universe.run()

    @staticmethod
    def get_plots():
        return {}

    @staticmethod
    def done():
        return False

    def step(self):
        print("[+] simulation time :", self.universe.get_time())
        print("[+] agents count :", len(self._agents_position))

    def _update_simulation(self):
        # get current agent's state
        agents = self.universe.get_agents(condition=self.agents_filter)
        agents_count = len(agents)

        # agents management
        self._agents_position = numpy.zeros((agents_count, 3))

        if self.should_draw_vertex():
            self._agents_vertex = numpy.zeros((agents_count, agents_count), dtype=bool)

            agent_index = {}
            for i, agent in enumerate(agents):
                agent_index[agent] = i

            for i, agent in enumerate(agents):
                children = agent.get_children()

                for child in children:
                    self._agents_vertex[i, agent_index[child]] = True

            self.set_vertex_on_agents(self._agents_vertex, agent_index)

        else:
            self._agents_vertex = None

        # chromo properties
        self._chromo_scale = numpy.zeros(agents_count)
        self._visual_radius = numpy.zeros(agents_count, dtype=float)
        self._visual_chromo = numpy.zeros((agents_count, 4))

        # get agents properties
        min_radius, max_radius = math.inf, -math.inf
        for i, agent in enumerate(agents):
            self._agents_position[i, :2] = agent.get_position()
            self._chromo_scale[i] = self.properties[Simulation.Properties.CHROMO](agent)
            self._chromo_scale[i] *= self.properties[Simulation.Properties.SCALE_CHROMO](agent)

            # get radius and extreme values
            self._visual_radius[i] = self.properties[Simulation.Properties.RADIUS](agent)

            if self._visual_radius[i] < min_radius:
                min_radius = self._visual_radius[i]

            if self._visual_radius[i] > max_radius:
                max_radius = self._visual_radius[i]

        # radius ticks
        ticks = (self.max_radius - self.min_radius)

        # extreme radius
        min_radius -= Simulation.EPSILON
        max_radius += Simulation.EPSILON

        # radius stride
        stride = (max_radius - min_radius) / ticks

        # re-scale radius
        self._visual_radius = (1 + self.min_radius + (self._visual_radius - min_radius) / stride).astype(int)

        # chromo scale
        r = numpy.fabs(self._chromo_scale * 6.0 - 3.0) - 1.0
        g = 2.0 - numpy.fabs((self._chromo_scale * 6.0 - 2.0))
        b = 2.0 - numpy.fabs(self._chromo_scale * 6.0 - 4.0)

        # red, green and blue
        self._visual_chromo[:, 0] = r
        self._visual_chromo[:, 1] = g
        self._visual_chromo[:, 2] = b
        self._visual_chromo[:, 3] = 1
