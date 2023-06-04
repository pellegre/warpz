from warpz.simulation.local import *
from models.flocker.land import *
from models.gene.pool import *
from models.dish.tallies import *

# bug radius
MAX_RADIUS = 25
MIN_RADIUS = 8

# grid size
GRID = 25

# energy seed
ENERGY_SEED = 1000 * GRID * GRID

# maximum motility
MAX_MOTILITY = 20

# current gene pool
common_gene_pool = {
    # energy of a new born bug
    Trait.BORN: (60, 120),

    # movement intensity
    Trait.MOTILITY: (9, MAX_MOTILITY),

    # bug inertia, resistance to motion change
    Trait.INERTIA: (120, 150),

    # bug color
    Trait.CHROMO: (0, GENE_CAPACITY),

    # how much more (from what I need) I'll eat - goloso
    Trait.GREEDINESS: (5, 15)
}


class FlockLand(Simulation):
    TOP_GENES = 10

    def __init__(self, warps_seed=ENERGY_SEED, grid=GRID):
        # tally cycle
        self.tally_cycle = 25

        # self._setup_case(warps_seed=warps_seed, grid=grid, tracking_type=Box.Tracking.ONE)
        # self._setup_case(warps_seed=warps_seed, grid=grid, tracking_type=Box.Tracking.QUAD)
        # self._setup_case(warps_seed=warps_seed, grid=grid, tracking_type=Box.Tracking.NINE)
        #
        # self._setup_case(warps_seed=warps_seed, grid=grid, tracking_type=Box.Tracking.QUAD,
        #                  agents_filter=lambda agent: isinstance(agent, Hawk), number=30)
        #
        # self._setup_case(warps_seed=300 * GRID * GRID, grid=grid, tracking_type=Box.Tracking.QUAD,
        #                  agents_filter=lambda agent: isinstance(agent, Hawk), number=50,
        #                  show_mesh=True, should_plot=True)
        #
        # self._setup_case(warps_seed=300 * GRID * GRID, grid=grid, tracking_type=Box.Tracking.QUAD,
        #                  agents_filter=lambda agent: isinstance(agent, Hawk), number=50,
        #                  show_mesh=True, should_plot=True, population_scale=300)
        #
        self._setup_case(warps_seed=300 * GRID * GRID, grid=grid, tracking_type=Box.Tracking.QUAD,
                         agents_filter=lambda agent: isinstance(agent, Hawk), number=50,
                         show_mesh=True, should_plot=True, population_scale=300, show_vertex=False)

        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.chromo if hasattr(agent, "chromo") else 0)

        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if hasattr(agent, "radius") else 0.5)
        self.set_property(Simulation.Properties.SCALE_CHROMO, 0.50)

        # plot hawks
        self.plots = {"hawks_count": 0.0, "dish_food": 0.0, "hawks_food": 0.0}

        # tallies
        self.bug_tally = AgentTally(agent_filter=lambda agent: isinstance(agent, Hawk))
        self.food_tally = CellTally(agent_tally=self.bug_tally, agent_filter=lambda agent: isinstance(agent, Food))
        self.collect_tally = CollectTally(self.dish, agent_tally=self.bug_tally, food_tally=self.food_tally)

        # periodic signaling for tallies
        self.dish.post(signal=self.collect_tally, period=self.tally_cycle)
        self.dish.post(signal=self.bug_tally, period=self.tally_cycle)
        self.dish.post(signal=self.food_tally, period=self.tally_cycle)

    def _setup_case(self, agents_filter=lambda agent: isinstance(agent, Agent),
                    number=10, show_mesh=False, should_plot=False, population_scale=None, show_vertex=True,
                    *args, **kwargs):

        self.dish = Dish(upper_bounds=[1, 1], bottom_bounds=[0, 0], *args, **kwargs)

        # show mesh
        self.show_mesh = show_mesh
        self.show_plot = should_plot
        self.show_vertex = show_vertex

        if population_scale:
            self.population_scale = population_scale
        else:
            self.population_scale = number

        # setup base case
        for i in range(0, number):
            # initial conditions
            x, y = random_state.uniform(-1, 1), random_state.uniform(0, 1)

            """ create common gene pool """
            gene_pool = GenePool(capacity=GENE_CAPACITY, genes=common_gene_pool)
            gene = gene_pool.get_random_gene(random_state)

            # instantiate bug
            hawk = Hawk(gene=gene, dish=self.dish, position=numpy.array([x, y]))
            self.dish.put(hawk)

        # init base class
        super().__init__(universe=self.dish, agents_filter=agents_filter, width=2900, height=1900,
                         min_radius=MIN_RADIUS, max_radius=MAX_RADIUS, plotter_period=1550)

    def get_plots(self):
        return self.plots

    def should_draw_vertex(self):
        return self.show_vertex

    def should_plot(self):
        return self.show_plot

    def set_vertex_on_agents(self, vertices, agent_index):
        signal_path = self.dish.get_signal_path()

        for source in signal_path:
            for target in signal_path[source]:
                if source in agent_index and target in agent_index:
                    max_distance = numpy.linalg.norm(self.get_upper_bounds() - self.get_bottom_bounds())
                    if numpy.linalg.norm(source.get_position() - target.get_position()) < \
                            0.50 * max_distance:
                        vertices[agent_index[source], agent_index[target]] = True

        self.dish.clear_signal_path()

    def get_mesh(self):
        if self.show_mesh:
            return super().get_mesh()
        return None

    @staticmethod
    def on_mouse_release(x, y, button):
        print(x, y)

    def done(self):
        return False

    def get_hawks_count(self):
        # warps in food and bugs
        hawks_count = 0.0
        for agent in self.dish.get_agents():
            if isinstance(agent, Hawk):
                hawks_count += 1.0

        return hawks_count

    def step(self):
        toc = time.process_time()

        # count cells
        hawks_count = self.get_hawks_count()
        self.plots["hawks_count"] = hawks_count / self.population_scale

        hawks_warps, food_warps = 0, 0
        for agent in self.dish.get_agents():
            if isinstance(agent, Hawk):
                hawks_warps += agent.warps
            elif isinstance(agent, Food):
                food_warps += agent.warps

        # total warps
        colony_warps = self.dish.warps + hawks_warps + food_warps
        self.plots["dish_food"] = food_warps / colony_warps
        self.plots["hawks_food"] = hawks_warps / colony_warps

        if self.dish.get_time() % self.tally_cycle == 0:
            # print some stats
            print("[.] ---- time          :", self.dish.get_time())

            universe_schwifties = self.dish.get_schwifties()
            agents_schwifties = sum([a.get_schwifties() for a in self.dish.get_agents()])

            print(f"[-] schwifty           : {agents_schwifties / universe_schwifties:.2f}")
            print("[-] current schwifties :",
                  sum([a.schwifty() for a in self.dish.get_agents()]) + self.dish.schwifty())

            print(f"[-] cost (average)     : {self.dish.get_average_cost():.2f}")

            # warps in food and bugs
            print("[w] hawks              :", hawks_count)
            print(f"[w] dish warps         : {self.dish.warps:.2f} ({self.dish.warps / colony_warps:.2f})")
            print(f"[w] bug warps          : {hawks_warps:.2f} ({hawks_warps / colony_warps:.2f})")
            print(f"[w] food warps         : {food_warps:.2f} ({food_warps / colony_warps:.2f})")
            print(f"[w] colony warps       : {colony_warps:.2f} ({colony_warps / colony_warps:.2f})")

            print("[@] stacked hawks      :", self.collect_tally.bug_stack)

            # get the top genes
            top_genes = {gene: count for gene, count in sorted(self.collect_tally.genes.items(),
                                                               key=lambda item: item[1]["count"], reverse=True)}

            # print top genes
            for i, gene in enumerate(top_genes):
                if gene in top_genes:
                    print("[@]", " {:04}".format(top_genes[gene]["count"]), gene,
                          "E = {:03.2f}".format(float(top_genes[gene]["warps"]) / float(top_genes[gene]["count"])))
                    if i == FlockLand.TOP_GENES:
                        break
