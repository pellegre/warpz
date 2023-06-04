import pandas

from scipy.stats import entropy
from functools import reduce

from warpz.simulation.local import *
from models.migrate.medium import *

standalone_run = False

# bug radius
MAX_RADIUS = 30
MIN_RADIUS = 5

# grid size
GRID = 10

# initial glucose seed
GLUCOSE_SEED = 700 * GRID * GRID

# reward scale
REWARD_SCALE = 0.10

# windows
if not standalone_run:
    from models.plotter.window import *

    # reward matrix
    reward_window = PlotMesh(windows=app.Window(990, 990), shape=(GRID, GRID))

    # glucose uptake matrix
    glucose_window = PlotMesh(windows=app.Window(990, 990), shape=(GRID, GRID))

    # mean free path windows
    population_window = Plotter(plotter_frame=app.Window(width=1140, height=460), title="population",
                                functions={"greedy", "prodigal"}, plotter_period=2000, log_scale=True)

    # delta (x/y) distribution
    MAX_DELTA = MOTILITY * MAX_MOTILITY
    delta_window = HistogramPlotter(plotter_frame=app.Window(width=1680, height=460),
                                    rows=1, cols=2, points=30, segment=[-MAX_DELTA, MAX_DELTA])

    @reward_window.windows.event
    def on_draw(dt):
        reward_window.windows.clear()

        reward_window.windows.set_position(0, 2400)

        reward_window.windows.set_title("reward distribution")

        reward_window.on_draw(dt)


    @glucose_window.windows.event
    def on_draw(dt):
        glucose_window.windows.clear()

        glucose_window.windows.set_position(0, 0)

        glucose_window.windows.set_title("glucose uptake")

        glucose_window.on_draw(dt)

    @delta_window.plotter.event
    def on_draw(dt):
        delta_window.plotter.set_position(2900, 1700)
        delta_window.plotter.set_title("delta (x/y) distribution")

        delta_window.plotter.clear()

        # visit on draw
        delta_window.on_draw(dt)

    @population_window.plotter.event
    def on_draw(dt):
        population_window.plotter.set_position(1120, 1700)
        population_window.on_draw(dt)


class TallyFrame:
    def __init__(self, genes):
        # genes frame
        self.genes_columns = ["time"] + [''] * len(genes)
        self.genes_index = {name: int(name) + 1 for name in genes}

        for each in self.genes_index:
            self.genes_columns[self.genes_index[each]] = str(each)

        self.genes_frame = pandas.DataFrame(columns=self.genes_columns)

        # observables frame
        self.observables_columns, self.observables_index = None, None

        # observables frame
        self.observables_frame = None

    def setup_observables_frame(self, values):
        if self.observables_frame is None:
            # observable columns
            self.observables_columns = ["time"] + list(values.keys())
            self.observables_index = {name: self.observables_columns.index(name) for name in self.observables_columns}

            # observables frame
            self.observables_frame = pandas.DataFrame(columns=self.observables_columns)

    def add_gene(self, step, genes):
        # set genes
        matrix = [[step] + list(g.get_traits()) for g in genes]
        frame = pandas.DataFrame(matrix, columns=self.genes_columns)

        # add gene
        self.genes_frame = pandas.concat([self.genes_frame, frame])

    def add_observables(self, step, values):
        # setup observables
        self.setup_observables_frame(values)

        # put observables
        row = [numpy.nan] * len(self.observables_columns)
        for each in values:
            row[self.observables_index[each]] = values[each]

        # set time
        row[self.observables_index["time"]] = step

        # create frame
        frame = pandas.DataFrame([row], columns=self.observables_columns)

        # add observables
        self.observables_frame = pandas.concat([self.observables_frame, frame])


class Tissue(Simulation):
    TOP_GENES = 5

    def __init__(self, knock_genes=None):
        # set knock genes
        if knock_genes is None:
            knock_genes = {Trait.MOTILITY: (1, 3), Trait.RADIUS: (150, GENE_CAPACITY)}

        # display cycle
        self.display_cycle = 10

        # gene pool
        # self.gene_pool = GenePopulation("./models/migrate/output/cut-06-12/", 370, knock_genes=knock_genes)
        self.gene_pool = GenePool(genes=releasing_wave_gene_pool, capacity=GENE_CAPACITY)

        # medium
        self.medium = Medium(upper_bounds=[0.40, 0.10], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.CLOSED,
                             tracking_type=Box.Tracking.QUAD, glucose=GLUCOSE_SEED, gene_pool=self.gene_pool)

        # accumulated reward (per cell unit)
        self.reward = numpy.zeros(2 * (GRID,), float)

        # accumulated glucose (per cell unit)
        self.glucose = numpy.zeros(2 * (GRID,), float)

        # accumulated glucose (per cell unit)
        self.cellula_count = numpy.ones(2 * (GRID,), int)

        # tallies
        self.tallies = TallyFrame(genes=self.gene_pool.get_genes())

        # draw vertex
        self.draw_vertex = False

        # instance blocks
        bottom, upper = self.medium.get_bottom_bounds(), self.medium.get_upper_bounds()
        # self.instance_block([(upper[0] + bottom[0]) / 8, (upper[1] + bottom[1]) / 2])
        # self.instance_block([7 * (upper[0] + bottom[0]) / 8, (upper[1] + bottom[1]) / 2])

        self.instance_tissue()

        balls = self.medium.get_agents(condition=lambda a: isinstance(a, Ball))
        min_ball_radius = min([ball.radius for ball in balls])
        max_ball_radius = max([ball.radius for ball in balls])

        stride = self.medium.get_upper_bounds() - self.medium.get_bottom_bounds()
        width, height = 2715, 1530
        windows_scale, simulation_scale = (width + height) / 2, (stride[0] + stride[1]) / 2

        if self.draw_vertex:
            min_radius = MIN_RADIUS
        else:
            min_radius = int(3 * windows_scale * (min_ball_radius / simulation_scale))

        max_radius = int(3 * windows_scale * (max_ball_radius / simulation_scale))

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        if not standalone_run:
            # balls radius
            self.set_property(Simulation.Properties.RADIUS,
                              lambda agent: agent.radius if isinstance(agent, Ball) else EPSILON)

            # color property
            self.set_property(Simulation.Properties.CHROMO,
                              lambda agent: agent.color if hasattr(agent, "color") else 0)

            if self.draw_vertex:
                super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Agent),
                                 width=width, height=height, min_radius=MIN_RADIUS, max_radius=max_radius)
            else:
                super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Ball),
                                 width=width, height=height, min_radius=min_radius, max_radius=max_radius)

    def should_draw_vertex(self):
        return self.draw_vertex

    def instance_block(self, position, factor=1):
        x_pos, y_pos = position[0], position[1]

        # single releasing plus rear cells
        for i in range(0, 40):
            x, y = random_state.uniform(x_pos - 0.03, x_pos - 0.01), random_state.uniform(y_pos - 0.03, y_pos - 0.01)
            gene = self.gene_pool.get_random_gene(random_state)

            self.medium.put(Cellulata.build_greedy(position=numpy.array([x, y]), gene=gene))

        for i in range(0, 5):
            gene = self.gene_pool.get_random_gene(random_state)
            self.medium.put(Cellulata.build_releasing(position=numpy.array([x_pos, y_pos]), gene=gene))

    def instance_tissue(self):
        # get bounds
        bottom, upper = self.medium.get_bottom_bounds(), self.medium.get_upper_bounds()

        # single releasing plus rear cells
        for i in range(0, 100):
            x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
            gene = self.gene_pool.get_random_gene(random_state)

            self.medium.put(Cellulata.build_greedy(position=numpy.array([x, y]), gene=gene))

            x, y = random_state.uniform(bottom[0], upper[0]), random_state.uniform(bottom[1], upper[1])
            gene = self.gene_pool.get_random_gene(random_state)

            self.medium.put(Cellulata.build_releasing(position=numpy.array([x, y]), gene=gene))

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.medium.get_mesh()

    def done(self):
        return False

    @staticmethod
    def print_population_top_genes(population, name):
        # get rear cells genes
        rear_genes = dict()
        for cell in population:
            if cell.gene not in rear_genes:
                rear_genes[cell.gene] = 1
            else:
                rear_genes[cell.gene] += 1

        # get the top genes
        top_genes = {gene: count for gene, count in sorted(rear_genes.items(),
                                                           key=lambda item: item[1], reverse=True)}

        # print top genes
        print("[@] --> " + name + " (top) genes")
        for i, gene in filter(lambda t: t[0] < Tissue.TOP_GENES, enumerate(top_genes)):
            print("[@]", " {:04}".format(top_genes[gene]), gene)

    def perform_surgery(self):
        for each in self.medium.get_agents(condition=lambda a: isinstance(a, Cell)):
            # wipe all chemokines
            # if (GRID // 3) - 2 < each.index[0] < (2 * GRID // 3) + 2:
            if (GRID // 3) - 2 < each.index[0]: #< (2 * GRID // 3) + 2:
                for chemokine in each.material:
                    if chemokine != GLUCOSE:
                        each.material[chemokine] = 0

                # remove cellulatas
                for cellulata in each.get_agents(condition=lambda a: isinstance(a, Cellulata)):
                    cellulata.apoptosis = True

                # clear stack
                each.clear_stack()

    def on_mouse_press(self, x, y, button):
        self.perform_surgery()

    def get_tissue_coverage(self, cellulatas):
        if len(cellulatas):
            # count of cells
            count = len(self.medium.get_cells())

            # get current density
            density, uniform = \
                [len(cell.get_children(condition=lambda a: isinstance(a, Cellulata) and a in cellulatas and
                                       a.get_cell() is cell)) / len(cellulatas) for cell in self.medium.get_cells()], \
                [1 / count for _ in self.medium.get_cells()]

            maximum_entropy = entropy(uniform)
            return entropy(density) / maximum_entropy
        else:
            # no coverage
            return 0

    def get_tissue_asymmetry(self, cellulatas):
        # calculate cellula area
        cells = self.medium.get_cells()

        # cellula per cell
        cellula_per_cell = max(1, (len(cellulatas) // len(cells)))

        # get covered area
        covered_cells = sum([
            abs(cellula_per_cell -
                len(cell.get_children(condition=lambda a: isinstance(a, Cellulata) and a.get_cell() is cell)))
            for cell in cells])

        # get coverage
        return covered_cells / (cellula_per_cell * len(cells))

    @staticmethod
    def get_persistence(cellulatas):
        # calculate cellula area
        return sum([cell.tallies.average()["cosine"]
                    for cell in cellulatas if "cosine" in cell.tallies.values]) / len(cellulatas)

    @staticmethod
    def get_genetic_observables(cellulatas):
        # get observables
        observables = [(cell.tallies.tallies["mutation-idv"] if "mutation-idv" in cell.tallies.tallies else 0,
                        cell.tallies.tallies["mutation-trt"] if "mutation-trt" in cell.tallies.tallies else 0,
                        cell.tallies.tallies["crossover-idv"] if "crossover-idv" in cell.tallies.tallies else 0)
                       for cell in cellulatas]

        # reduce observables
        return reduce(lambda o, p: (o[0] + p[0], o[1] + p[1], o[2] + p[2]), observables)

    def collect_mesh_observables(self, cellulatas):
        # go over each cellula
        for each in cellulatas:
            # get cell
            cell = each.get_cell()

            # accumulate reward
            self.reward[cell.index[1], cell.index[0]] += each.reward

            # accumulate glucose
            self.glucose[cell.index[1], cell.index[0]] += each.glucose

            # count cellula
            self.cellula_count[cell.index[1], cell.index[0]] += 1

    def reset_mesh_observables(self):
        # reset mesh counters
        self.reward = numpy.zeros(2 * (GRID,), float)
        self.glucose = numpy.zeros(2 * (GRID,), float)
        self.cellula_count = numpy.ones(2 * (GRID,), int)

    def collect_tallies(self):
        # greedy and releasing
        greedy = list(self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_greedy()))
        releasing = list(self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_releasing()))

        # cellulata
        cellulatas = greedy + releasing

        # new born genes
        self.tallies.add_gene(self.medium.get_time(), [cell.gene for cell in cellulatas
                                                       if self.medium.get_time() - 1 == cell.born_time])

        # get observables
        coverage = self.get_tissue_coverage(cellulatas)
        coverage_greedy = self.get_tissue_coverage(greedy)
        coverage_releasing = self.get_tissue_coverage(releasing)
        persistence = self.get_persistence(cellulatas)
        asymmetry = self.get_tissue_asymmetry(cellulatas)

        # get genetic stats
        mutation_individual, mutation_trait, crossover_individual = self.get_genetic_observables(cellulatas)

        # chemokines
        cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))
        releasing_chemokines = sum([cell.material[CHEMOKINE] for cell in cells])

        # glucose distribution
        free_glucose = sum([cell.material[GLUCOSE] for cell in cells])
        releasing_glucose = sum([cell.glucose for cell in releasing])
        greedy_glucose = sum([cell.glucose for cell in greedy])

        # add tallies
        self.tallies.add_observables(self.medium.get_time(), {
            "releasing-population": len(releasing), "greedy-population": len(greedy), "coverage": coverage,
            "persistence": persistence, "greedy-glucose": releasing_glucose, "releasing-chemokine": releasing_chemokines,
            "releasing-glucose": greedy_glucose, "free-glucose": free_glucose, "asymmetry": asymmetry,
            "crossover-individual": crossover_individual, "mutation-individual": mutation_individual,
            "mutation-trait": mutation_trait, "coverage-greedy": coverage_greedy,
            "coverage-releasing": coverage_releasing,
        })

    def collect_delta_distribution(self):
        cellulatas = self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata))

        for cell in cellulatas:
            if len(cell.tallies.values) > 0 and "delta_x" in cell.tallies.values:
                if math.fabs(cell.tallies.values["delta_x"]) < MAX_DELTA and \
                        math.fabs(cell.tallies.values["delta_y"]) < MAX_DELTA:
                    delta_window.add([[cell.tallies.values["delta_x"], cell.tallies.values["delta_y"]]])

    def get_mean_free_path_ratio(self, cellulatas):
        return sum([numpy.linalg.norm(cell.get_position() - cell.initial_position) /
                    (math.sqrt(self.medium.get_time() - cell.born_time) * cell.tallies.average()["distance"])
                    for cell in cellulatas if "distance" in cell.tallies.values]) / len(cellulatas)

    def step(self):
        # get cellulatas
        greedy = set(self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_greedy()))
        releasing = set(self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata) and a.is_releasing()))
        cellulatas = greedy.union(releasing)

        coverage = self.get_tissue_coverage(cellulatas)
        if self.get_tissue_coverage(cellulatas) >= 0.98:
            self.perform_surgery()

        # print some stats
        if self.medium.get_time() % self.display_cycle == 0:
            # collect mesh observables
            self.collect_mesh_observables(cellulatas)

            # collect delta distribution
            self.collect_delta_distribution()

            # normalize reward
            self.reward /= self.cellula_count
            self.reward[self.reward < -REWARD_SCALE] = -1
            self.reward[self.reward > REWARD_SCALE] = 1
            self.reward[(self.reward < REWARD_SCALE) & (self.reward > -REWARD_SCALE)] = 0

            # update mesh plots
            reward_window.set_mesh(self.reward, -1, 1)
            glucose_window.set_mesh(self.glucose / self.cellula_count)

            # reset observables
            self.reset_mesh_observables()

            # mean free path ratio
            population_window.update_data(self.medium.get_time(), {"greedy": len(greedy),
                                                                   "prodigal": len(releasing)})

            # chemokines (in unit cells)
            cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))
            releasing_chemokines = sum([cell.material[CHEMOKINE] for cell in cells])

            # glucose distribution
            free_glucose = sum([cell.material[GLUCOSE] for cell in cells])
            releasing_glucose = sum([cell.glucose for cell in releasing])
            greedy_glucose = sum([cell.glucose for cell in greedy])

            # total glucose
            total_glucose = self.medium.glucose + free_glucose + releasing_glucose + greedy_glucose

            if len(greedy) and len(releasing):
                print("-------------- (step) ", self.medium.get_time())
                universe_schwifties = self.medium.get_schwifties()
                agents_schwifties = sum([a.get_schwifties() for a in self.medium.get_agents()])

                print(f"[-] schwifty         : {agents_schwifties / universe_schwifties:.2f}")

                print("[#] --> glucose      :", total_glucose)
                print("[#] greedy           :", greedy_glucose)
                print("[#] releasing        :", releasing_glucose)
                print("[#] free             :", free_glucose)
                print("[#] metabolized      :", self.medium.glucose)

                print("[#] --> greedy       :", len(greedy))
                print("[#] chemokines (max) :", max(greedy, key=lambda a: a.chemokines).chemokines)
                print("[#] chemokines (min) :", min(greedy, key=lambda a: a.chemokines).chemokines)

                print("[#] --> releasing    :", len(releasing))
                print("[#] chemokines (max) :", max(releasing, key=lambda a: a.chemokines).chemokines)
                print("[#] chemokines (min) :", min(releasing, key=lambda a: a.chemokines).chemokines)

                print("[#] releasing (chem) :", releasing_chemokines)

                # get genetic stats
                mutation_individual, mutation_trait, crossover_individual = self.get_genetic_observables(cellulatas)
                print("[#] crossover (ind)  : ", crossover_individual)
                print("[#] mutation  (ind)  : ", mutation_individual)
                print("[#] mutation (trait) : ", mutation_trait)
                print("[#] coverage         : ", coverage)

                # get rear cells genes
                self.print_population_top_genes(greedy, "greedy")
                self.print_population_top_genes(releasing, "releasing")

