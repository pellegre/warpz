import pandas
from scipy.stats import entropy

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
target_correlation = PlotMesh(windows=app.Window(905, 905), shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))
cross_correlation = PlotMesh(windows=app.Window(905, 905), shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))

# bipolarity sensor level
BIPOLAR_LEVEL = 30
bipolar_level = PlotMesh(windows=app.Window(1015, 995), shape=(BIPOLAR_LEVEL, BIPOLAR_LEVEL))

# signals tiles
signals = TilesPlotter(plotter_frame=app.Window(width=1015, height=995), rows=2 * NUMBER_OF_UNITS + 2, cols=1,
                       points=TIME_SCALE, time_factor=TIME_FACTOR)

# signals tiles
q_function = TilesPlotter(plotter_frame=app.Window(width=1210, height=1080), rows=NUMBER_OF_UNITS, cols=4,
                          points=NUMBER_OF_STATES)


@signals.plotter.event
def on_draw(dt):
    q_function.plotter.set_position(1140, 0)
    signals.plotter.set_title("signal spikes (time)")

    signals.plotter.clear()

    signals.on_draw(dt)


@bipolar_level.windows.event
def on_draw(dt):
    bipolar_level.windows.clear()

    bipolar_level.windows.set_position(0, 0)

    bipolar_level.windows.set_title("bipolar level")

    bipolar_level.on_draw(dt)


@q_function.plotter.event
def on_draw(dt):
    signals.plotter.set_position(0, 2400)

    q_function.plotter.set_title("Q function (catcher | target) (idle | move)")

    q_function.plotter.clear()

    q_function.on_draw(dt)


@catcher_correlation.windows.event
def on_draw(dt):
    catcher_correlation.windows.clear()

    catcher_correlation.windows.set_position(3400, 2400)

    catcher_correlation.windows.set_title("correlation (catcher)")

    catcher_correlation.on_draw(dt)


@target_correlation.windows.event
def on_draw(dt):
    target_correlation.windows.clear()

    target_correlation.windows.set_position(2050, 2400)

    target_correlation.windows.set_title("correlation (target)")

    target_correlation.on_draw(dt)


@cross_correlation.windows.event
def on_draw(dt):
    cross_correlation.windows.clear()

    cross_correlation.windows.set_position(1150, 2400)

    cross_correlation.windows.set_title("cross correlation")

    cross_correlation.on_draw(dt)


class SynchroGame(Simulation):
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

        # action frame columns
        self.units_columns = ["catcher-" + str(i) for i in range(0, len(self.catcher.units))] + \
                             ["target-" + str(i) for i in range(0, len(self.target.units))]

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

    def step(self):
        # catcher / target actions
        catcher_action = [player.action_taken for player in self.catcher.units]
        target_action = [player.action_taken for player in self.target.units]

        # actions frame (taken at each time step per player)
        actions = catcher_action + target_action

        # accumulate on action frame
        action_frame = pandas.DataFrame([[self.get_time()] + actions], columns=["time"] + self.units_columns)
        self.action_frame = pandas.concat([self.action_frame, action_frame])

        # plot action tiles
        action_tiles = [[int(x)] if x is not None else [0] for x in catcher_action + [0, 0] + target_action]
        signals.add(self.get_time(), action_tiles)

        # plot Q matrix
        q_matrix = numpy.array([[c.q_matrix[:, 0] / numpy.fabs(c.q_matrix[:, 0]).max()] +
                                [t.q_matrix[:, 0] / numpy.fabs(t.q_matrix[:, 0]).max()] +
                                [c.q_matrix[:, 1] / numpy.fabs(c.q_matrix[:, 1]).max()] +
                                [t.q_matrix[:, 1] / numpy.fabs(t.q_matrix[:, 1]).max()]
                                for c, t in zip(self.catcher.units, self.target.units)])
        q_function.push(q_matrix)

        # catcher / target units
        catcher = ["catcher-" + str(i) for i in range(0, len(self.catcher.units))]
        target = ["target-" + str(i) for i in range(0, len(self.target.units))]

        # action correlation matrix
        catcher_matrix = self.get_action_correlation(self.action_frame[catcher], self.action_frame[catcher]).values
        target_matrix = self.get_action_correlation(self.action_frame[target], self.action_frame[target]).values
        cross_matrix = self.get_action_correlation(self.action_frame[target], self.action_frame[catcher]).values

        # plot data
        catcher_correlation.set_mesh(catcher_matrix)
        target_correlation.set_mesh(target_matrix)
        cross_correlation.set_mesh(cross_matrix)

        # print some stats
        if self.medium.get_time() % self.display_cycle == 0:
            print("[+] (step) ", self.medium.get_time())


class MovableTarget(SynchroGame):
    def __init__(self):
        # medium
        medium = Medium(upper_bounds=[1.0, 1.0], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.CLOSED,
                        tracking_type=Box.Tracking.QUAD, glucose=0)

        # target
        target = Target(position=numpy.array([0.70, 0.70]), units_number=NUMBER_OF_UNITS,
                        motility=512 * MOTILITY, states_number=NUMBER_OF_STATES)

        # catcher
        catcher = Catcher(position=numpy.array([0.55, 0.55]), units_number=NUMBER_OF_UNITS,
                          motility=512 * MOTILITY, states_number=NUMBER_OF_STATES)

        # setup pointers
        medium.put(target)
        medium.put(catcher)

        # synchro game
        super(MovableTarget, self).__init__(medium=medium, target=target, catcher=catcher)

    def step(self):
        # get distance
        distance = numpy.linalg.norm(self.target.get_position() - self.catcher.get_position())

        # check for collisions
        if distance < 0.025:
            # set random positions
            x, y = random_state.uniform(0, 1), random_state.uniform(0, 1)
            setattr(self.target, Agent.POSITION, numpy.array([x, y]))
        else:
            super(MovableTarget, self).step()


class ChemokineTarget(SynchroGame):

    @staticmethod
    def deploy_greedy(position):
        # get a gene
        gene = seed_pool.get_random_gene(random_state)

        # automata genes
        gene.set_trait(Trait.EPSILON, 120)
        gene.set_trait(Trait.ALPHA, 200)
        gene.set_trait(Trait.GAMMA, 200)
        gene.set_trait(Trait.STATES, NUMBER_OF_STATES)
        gene.set_trait(Trait.UNITS, NUMBER_OF_UNITS)
        gene.set_trait(Trait.RADIUS, GENE_CAPACITY)

        # consumption and production rate
        gene.set_trait(Trait.CONSUMPTION, GENE_CAPACITY)
        gene.set_trait(Trait.BIPOLARITY, 8096 * 8096)
        gene.set_trait(Trait.RELEASE, 30)
        gene.set_trait(Trait.PRODUCTION, 0)

        # cellulata instance
        cellulata = Cellulata.build(replication=False, position=numpy.array(position), gene=gene)
        cellulata.motility = 78 * MOTILITY

        # return cellulata
        return cellulata

    def __init__(self):
        # medium
        medium = Medium(upper_bounds=[1.0, 1.0], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.OPEN,
                        tracking_type=Box.Tracking.QUAD, glucose=8096 * 8096)

        # target
        target = Leader(position=numpy.array([0.60, 0.60]), units_number=NUMBER_OF_UNITS,
                        direction=numpy.array([1.0, 0.0]), motility=48 * MOTILITY, states_number=NUMBER_OF_STATES,
                        releasing_mode=True)

        target.gene.set_trait(Trait.RELEASE, 100)

        # catcher
        catcher = self.deploy_greedy(position=numpy.array([0.55, 0.55]))

        # setup pointers
        medium.put(target)
        medium.put(catcher)

        # synchro game
        super(ChemokineTarget, self).__init__(medium=medium, target=target, catcher=catcher)

    def step(self):
        # run simulation step
        super().step()

        # set random direction
        if self.get_time() % 200 == 0:
            theta = numpy.random.uniform(0, 2 * math.pi)
            self.target.movement = numpy.array([math.cos(theta), math.sin(theta)])


class BipolarDynamics(SynchroGame):

    def deploy_releasing(self, position):
        # get a gene
        gene = self.gene_pool.get_random_gene(random_state)

        # automata genes
        gene.set_trait(Trait.EPSILON, 120)
        gene.set_trait(Trait.ALPHA, 200)
        gene.set_trait(Trait.GAMMA, 200)
        gene.set_trait(Trait.STATES, NUMBER_OF_STATES)
        gene.set_trait(Trait.UNITS, NUMBER_OF_UNITS)
        gene.set_trait(Trait.RADIUS, GENE_CAPACITY)

        # consumption and production rate
        gene.set_trait(Trait.CONSUMPTION, GENE_CAPACITY)
        gene.set_trait(Trait.BIPOLARITY, GENE_CAPACITY // 4)
        gene.set_trait(Trait.RELEASE, 30)
        gene.set_trait(Trait.PRODUCTION, 7)

        # cellulata instance
        bipolar_threshold = BIPOLARITY_FACTOR * gene.get_trait(Trait.BIPOLARITY) * gene.get_trait(Trait.CONSUMPTION)
        cellulata = Cellulata.build(replication=False, position=numpy.array(position), gene=gene,
                                    chemokines=bipolar_threshold)
        cellulata.motility = 128 * MOTILITY

        # return cellulata
        return cellulata

    def deploy_greedy(self, position):
        # get a gene
        gene = self.gene_pool.get_random_gene(random_state)

        # automata genes
        gene.set_trait(Trait.EPSILON, 120)
        gene.set_trait(Trait.ALPHA, 200)
        gene.set_trait(Trait.GAMMA, 200)
        gene.set_trait(Trait.STATES, NUMBER_OF_STATES)
        gene.set_trait(Trait.UNITS, NUMBER_OF_UNITS)
        gene.set_trait(Trait.RADIUS, GENE_CAPACITY)

        # consumption and production rate
        gene.set_trait(Trait.CONSUMPTION, GENE_CAPACITY)
        gene.set_trait(Trait.BIPOLARITY, GENE_CAPACITY // 4)
        gene.set_trait(Trait.RELEASE, 30)
        gene.set_trait(Trait.PRODUCTION, 7)

        # cellulata instance
        cellulata = Cellulata.build(replication=False, position=numpy.array(position), gene=gene)
        cellulata.motility = 128 * MOTILITY

        # return cellulata
        return cellulata

    def __init__(self):
        # medium
        medium = Medium(upper_bounds=[1.0, 1.0], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.OPEN,
                        tracking_type=Box.Tracking.QUAD, glucose=8096 * 8096)

        # gene pool
        self.gene_pool = GenePool(genes=seed_gene_pool, capacity=GENE_CAPACITY)

        # target
        target = self.deploy_releasing(position=numpy.array([0.55, 0.55]))

        # catcher
        catcher = self.deploy_greedy(position=numpy.array([0.30, 0.30]))

        # setup pointers
        medium.put(target)
        medium.put(catcher)

        # bipolarity leve
        self.bipolar_level = -1 * numpy.ones((BIPOLAR_LEVEL, BIPOLAR_LEVEL))
        self.bipolar_threshold = BIPOLARITY_FACTOR * catcher.gene.get_trait(Trait.BIPOLARITY) * \
                                 catcher.gene.get_trait(Trait.CONSUMPTION)

        # synchro game
        super(BipolarDynamics, self).__init__(medium=medium, target=target, catcher=catcher)

    def step(self):
        # run simulation step
        super().step()

        # reset levels
        self.bipolar_level[:, :] = -1

        # target
        idx = int(min(BIPOLAR_LEVEL - 1, BIPOLAR_LEVEL * (self.target.chemokines / self.bipolar_threshold)))
        self.bipolar_level[0:idx, 1:BIPOLAR_LEVEL // 2 - 1] = 1

        # catcher
        idx = int(min(BIPOLAR_LEVEL - 1, BIPOLAR_LEVEL * (self.catcher.chemokines / self.bipolar_threshold)))
        self.bipolar_level[0:idx, BIPOLAR_LEVEL // 2 + 1:BIPOLAR_LEVEL - 1] = 1

        # update bipolarity level
        bipolar_level.set_mesh(self.bipolar_level)
