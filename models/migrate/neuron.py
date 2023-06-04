import pandas

from warpz.simulation.local import *

from models.migrate.medium import *
from models.plotter.window import *

# bug radius
MAX_RADIUS = 30
MIN_RADIUS = 30

# grid size
GRID = 10

# rolling time for graphs
ROLLING_TIME = 400

# number of units
NUMBER_OF_UNITS = 8

# number of units
NUMBER_OF_STATES = 16

# time scale
TIME_SCALE, TIME_FACTOR = 20000, 100

# catcher / target action correlation matrix
catcher_correlation = PlotMesh(windows=app.Window(905, 905), shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))

# signals tiles
signals = TilesPlotter(plotter_frame=app.Window(width=2220, height=1080), rows=NUMBER_OF_UNITS, cols=1,
                       points=TIME_SCALE, time_factor=TIME_FACTOR)

# signals tiles
q_function = TilesPlotter(plotter_frame=app.Window(width=1220, height=905), rows=NUMBER_OF_UNITS, cols=2,
                          points=NUMBER_OF_STATES)

# cooperation windows
cooperation = Plotter(plotter_frame=app.Window(width=1575, height=905), title="cooperation",
                      functions={"cooperation"}, plotter_period=1024)


@signals.plotter.event
def on_draw(dt):
    signals.plotter.set_position(0, 0)
    signals.plotter.set_title("signal spikes (time)")

    signals.plotter.clear()

    signals.on_draw(dt)


@q_function.plotter.event
def on_draw(dt):
    q_function.plotter.set_position(0, 2400)
    q_function.plotter.set_title("Q function (idle | move)")

    q_function.plotter.clear()

    q_function.on_draw(dt)


@catcher_correlation.windows.event
def on_draw(dt):
    catcher_correlation.windows.clear()

    catcher_correlation.windows.set_position(1355, 2400)

    catcher_correlation.windows.set_title("correlation (catcher)")

    catcher_correlation.on_draw(dt)


@cooperation.plotter.event
def on_draw(dt):
    cooperation.plotter.set_position(2400, 2400)
    cooperation.on_draw(dt)


class NeuronGame(Simulation):
    def __init__(self, medium, catcher, target):
        # set medium
        self.medium = medium

        # catcher and target
        self.catcher, self.target = catcher, target

        # display cycle
        self.display_cycle = 100

        # draw vertex
        self.draw_vertex = False

        balls = self.medium.get_agents(condition=lambda a: isinstance(a, Ball))
        min_ball_radius = min([ball.radius for ball in balls])
        max_ball_radius = max([ball.radius for ball in balls])

        stride = self.medium.get_upper_bounds() - self.medium.get_bottom_bounds()
        width, height = 1495, 1080
        windows_scale, simulation_scale = (width + height) / 2, (stride[0] + stride[1]) / 2

        if self.draw_vertex:
            min_radius = MIN_RADIUS
        else:
            min_radius = int(20 * windows_scale * (min_ball_radius / simulation_scale))

        max_radius = int(20 * windows_scale * (max_ball_radius / simulation_scale))

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        # balls radius
        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if isinstance(agent, Ball) else EPSILON)

        # color property
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.color if hasattr(agent, "color") else 0)

        # get distance
        self.distance = numpy.linalg.norm(self.target.get_position() - self.catcher.get_position())

        # action frame columns
        self.units_columns = ["catcher-" + str(i) for i in range(0, len(self.catcher.units))] + ["success"]

        # action frame
        self.action_frame = pandas.DataFrame(columns=["time"] + self.units_columns)

        if self.draw_vertex:
            super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Agent),
                             width=width, height=height, min_radius=MIN_RADIUS, max_radius=max_radius)
        else:
            super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Ball),
                             width=width, height=height, min_radius=min_radius, max_radius=max_radius)

    def should_draw_vertex(self):
        return self.draw_vertex

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.medium.get_mesh()

    def done(self):
        return False

    @staticmethod
    def get_action_correlation(one: pandas.DataFrame, other: pandas.DataFrame):
        rows = other.columns
        columns = one.columns

        matrix = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)
        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                one_data, other_data = one[col].tail(ROLLING_TIME).values.astype(float), \
                                       other[row].tail(ROLLING_TIME).values.astype(float)

                n = len(one_data)

                z = numpy.sum((one_data - numpy.average(one_data)) * (other_data - numpy.average(other_data)))
                x = (n - 1) * numpy.std(one_data) * numpy.std(other_data)
                matrix[i, j] = z / x

        return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                                columns=[i + 1 for i in range(0, len(columns))])

    @staticmethod
    def get_success_rate(frame: pandas.DataFrame):
        columns = [c for c in frame.columns if c != "success"]

        # success (closer to target)
        success = frame["success"].values

        # define matrix
        matrix = numpy.zeros(shape=(len(columns), len(columns)), dtype=float)

        for i, row in enumerate(columns):
            for j, col in enumerate(columns):
                one_data, other_data = frame[col].values, frame[row].values

                z = numpy.sum((one_data * other_data) * success)
                x = numpy.sum(one_data * other_data)

                if x > 0:
                    matrix[i, j] = z / x

        matrix /= numpy.sum(matrix[:, :])
        return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                                columns=[i + 1 for i in range(0, len(columns))])

    @staticmethod
    def get_cooperation(action_correlation, success_rate):
        # upper indices
        idx = numpy.triu_indices(action_correlation.values.shape[0], 1)

        # get values
        ac, sr = action_correlation.values[idx], success_rate.values[idx]
        np = len(ac)

        # get cooperation
        coop = (1 / (np - 1)) * numpy.sum((ac - numpy.mean(ac)) * (sr - numpy.mean(sr)))
        if coop is not numpy.nan:
            return coop
        return 0

    def step(self):
        # get distance
        distance = numpy.linalg.norm(self.target.get_position() - self.catcher.get_position())

        # check for collisions
        if distance < 0.025:
            # set random positions
            x, y = random_state.uniform(0, 1), random_state.uniform(0, 1)
            setattr(self.target, Agent.POSITION, numpy.array([x, y]))
        else:
            # get success
            success = 1 if (distance - self.distance) < 0 else 0

            # catcher / target actions
            catcher_action = [player.action_taken for player in self.catcher.units] + [success]

            # update distance
            self.distance = distance

            # accumulate on action frame
            action_frame = pandas.DataFrame([[self.get_time()] + catcher_action], columns=["time"] + self.units_columns)
            self.action_frame = pandas.concat([self.action_frame, action_frame])

            # plot action tiles
            action_tiles = [[int(x)] if x is not None else [0] for x in catcher_action]
            signals.add(self.get_time(), action_tiles)

            # plot Q matrix
            q_matrix = numpy.array([[c.q_matrix[:, 0] / numpy.fabs(c.q_matrix[:, 0]).max()] +
                                    [c.q_matrix[:, 1] / numpy.fabs(c.q_matrix[:, 1]).max()]
                                    for c in self.catcher.units])
            q_function.push(q_matrix)

            # catcher / target units
            catcher = ["catcher-" + str(i) for i in range(0, len(self.catcher.units))]

            # action correlation matrix
            action_correlation = self.get_action_correlation(self.action_frame[catcher], self.action_frame[catcher])
            success_rate = self.get_success_rate(self.action_frame[catcher + ["success"]])

            # cooperation
            if self.get_time() > 50:
                cooperation.update_data(self.get_time(),
                                        {"cooperation": 10E5 * self.get_cooperation(action_correlation, success_rate)})

            # plot data
            catcher_correlation.set_mesh(action_correlation.values)

            # print some stats
            if self.medium.get_time() % self.display_cycle == 0:
                print("[+] (step) ", self.medium.get_time())


class FixedTarget(NeuronGame):
    def __init__(self):
        # medium
        medium = Medium(upper_bounds=[1.0, 1.0], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.CLOSED,
                        tracking_type=Box.Tracking.QUAD, glucose=0)

        # target
        target = Target(position=numpy.array([0.70, 0.70]), units_number=NUMBER_OF_UNITS,
                        motility=0, states_number=NUMBER_OF_STATES)

        # catcher
        catcher = Catcher(position=numpy.array([0.10, 0.10]), units_number=NUMBER_OF_UNITS,
                          motility=512 * MOTILITY, states_number=NUMBER_OF_STATES)

        # setup pointers
        medium.put(target)
        medium.put(catcher)

        # synchro game
        super(FixedTarget, self).__init__(medium=medium, target=target, catcher=catcher)

