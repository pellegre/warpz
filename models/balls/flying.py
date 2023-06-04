from warpz.simulation.local import *
from warpz.space.box import *

import time
import functools
import numpy

# random state
random_state = numpy.random.RandomState()

# bug radius
MAX_RADIUS = 25
MIN_RADIUS = 8

# grid size
GRID = 10


class Parcel(Cell):
    def __init__(self, **kwargs):
        # initial stack of energy
        self.light, self.initial = 0, 0
        super().__init__(**kwargs)


# ===========
#
# Land (with a parcel mesh)
#
# ===========

class Land(Box):
    def __init__(self, grid=10, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Parcel, **kwargs)

        # food and energy release
        self.mesh = numpy.zeros(self._dimensions * (self._grid,), float)

    def get_mesh(self):
        # mesh
        for each in self.get_children(condition=lambda c: isinstance(c, Parcel)):
            self.mesh[each.index[1], each.index[0]] = each.light

        # get light mesh
        return self.mesh


# ===========
#
# Flying Balls
#
# ===========

class FlyingBall(Ball):
    def __init__(self, motility=0.005, **kwargs):
        # ball motility
        self.motility = motility
        self.chromo = 0.1

        # inertial ball counter
        self.movement_counter, self.inertia = 0, 100

        # random ball direction
        self.direction = self.get_random_direction()

        # init base ball class
        super().__init__(**kwargs)

    def set_direction(self, direction):
        # ball inertia
        if self.movement_counter >= self.inertia:
            # set new ball direction
            self.direction = direction

            # reset ball inertial counter
            self.movement_counter = 0

    @staticmethod
    def get_random_direction():
        # random ball direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def transport(self):
        # ball transport
        yield Transport(self.motility, self.direction)

        # new ball time step
        self.movement_counter += 1

    def move(self):
        # attempt to move the ball, after inertial ball property
        if self.movement_counter >= self.inertia:
            # random direction
            self.direction = self.get_random_direction()
            self.movement_counter = 0

        # ball transportation
        yield self.transport()

    def interact(self, signals):
        # move my ball
        yield self.move()


# ===========
#
# Ball Pointer
#
# ===========

# chemical queue
class ThereIsOne(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, ball):
        # get source
        who = self.get_source()

        # ball direction
        who.direction = ball.get_position() - who.get_position()
        who.direction /= numpy.linalg.norm(who.direction)


class BallPointer(Ball):
    def __init__(self, motility=0.005, **kwargs):
        # motility
        self.motility = motility
        self.chromo = 0.7

        # random direction (at first)
        self.direction = FlyingBall.get_random_direction()

        # init base class
        super().__init__(**kwargs)

    def transport(self):
        # ball transport
        yield Transport(self.motility, self.direction)

    @staticmethod
    def point():
        # ball transport
        yield ThereIsOne(target=lambda c: isinstance(c, FlyingBall))

    def interact(self, signals):
        # sense balls
        yield self.point()

        # follow ball
        yield self.transport()


class Yard(Simulation):
    def __init__(self):
        # tally cycle
        self.tally_cycle = 100

        # land on the yard
        self.land = Land(grid=GRID, boundary=Box.Boundary.CLOSED,
                         upper_bounds=[0.1, 0.1], bottom_bounds=[0.0, 0.0])

        bottom, upper = self.land.get_bottom_bounds(), self.land.get_upper_bounds()

        # a flying ball
        for i in range(0, 50):
            x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
            self.land.put(FlyingBall(radius=0.001, motility=0.0002, position=numpy.array([x, y])))

        x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
        self.land.put(BallPointer(radius=0.005, motility=0.001, position=numpy.array([x, y])))

        x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
        self.land.put(BallPointer(radius=0.005, motility=0.001, position=numpy.array([x, y])))

        x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
        self.land.put(BallPointer(radius=0.005, motility=0.001, position=numpy.array([x, y])))

        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.chromo if hasattr(agent, "chromo") else 0)

        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if isinstance(agent, Ball) else EPSILON)

        balls = self.land.get_agents(condition=lambda a: isinstance(a, Ball))
        min_ball_radius = min([ball.radius for ball in balls])
        max_ball_radius = max([ball.radius for ball in balls])

        stride = self.land.get_upper_bounds() - self.land.get_bottom_bounds()
        width, height = 2800, 1900
        windows_scale, simulation_scale = (width + height) / 2, (stride[0] + stride[1]) / 2

        min_radius = int(5 * windows_scale * (min_ball_radius / simulation_scale))
        max_radius = int(5 * windows_scale * (max_ball_radius / simulation_scale))

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        # init base class
        super().__init__(universe=self.land, width=width, height=height,
                         agents_filter=lambda agent: isinstance(agent, Agent), min_radius=MIN_RADIUS,
                         max_radius=max_radius, plotter_period=1550)

    def should_draw_vertex(self):
        return True

    def should_plot(self):
        return None

    def get_mesh(self):
        return None

    @staticmethod
    def on_mouse_release(x, y, button):
        print(x, y)

    def done(self):
        return False

    def step(self):
        # print some stats
        print("[.] ---- time :", self.land.get_time())

        universe_schwifties = self.land.get_schwifties()
        agents_schwifties = sum([a.get_schwifties() for a in self.land.get_agents()])

        print(f"[-] schwifty           : {agents_schwifties / universe_schwifties:.2f}")

