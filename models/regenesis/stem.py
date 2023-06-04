from warpz.simulation.local import *
from models.regenesis.tissue import *
from models.gene.pool import *
from models.dish.tallies import *

# bug radius
MAX_RADIUS = 14
MIN_RADIUS = 8

# grid size
GRID = 10

# energy seed
ENERGY_SEED = 1800 * GRID * GRID

# maximum motility
MAX_MOTILITY = 9

# current gene pool
common_gene_pool = {
    # energy of a new born bug
    Trait.BORN: (60, 120),

    # movement intensity
    Trait.MOTILITY: (3, MAX_MOTILITY),

    # bug inertia, resistance to motion change
    Trait.INERTIA: 50,

    # resistance to bites
    Trait.MUTATION: 60,

    # signaling rate
    Trait.SIGNALING: (30, 50),

    # how much more (from what I need) I'll eat - goloso
    Trait.GREEDINESS: (5, 15)
}

gene_fixed_amplification_pool = copy.deepcopy(common_gene_pool)
gene_fixed_amplification_pool[Trait.AMPLIFICATION] = int(GENE_CAPACITY / 2)

gene_varying_amplification_pool = copy.deepcopy(common_gene_pool)
gene_varying_amplification_pool[Trait.AMPLIFICATION] = (0, GENE_CAPACITY)


class Tissue(Simulation):
    CELL_SCALE = 900
    TOP_GENES = 10

    def __init__(self, tally_cycle=100, energy_seed=ENERGY_SEED, grid=GRID, setup=None):
        # tally cycle
        self.tally_cycle = tally_cycle

        # dish
        self.dish = Dish(warps_seed=energy_seed, grid=grid)

        # chromo agent
        self.set_property(Simulation.Properties.CHROMO, lambda agent: agent.chromo)
        self.set_property(Simulation.Properties.RADIUS, lambda agent: agent.radius)

        # setup case
        self.setup_case_two_with_chalones_varying_amplification()

        # tallies
        self.bug_tally = AgentTally(agent_filter=lambda agent: isinstance(agent, Cellulata))
        self.food_tally = CellTally(agent_tally=self.bug_tally, agent_filter=lambda agent: isinstance(agent, Food))
        self.collect_tally = CollectTally(self.dish, agent_tally=self.bug_tally, food_tally=self.food_tally)

        # periodic signaling for tallies
        self.dish.post(signal=self.collect_tally, period=self.tally_cycle)
        self.dish.post(signal=self.bug_tally, period=self.tally_cycle)
        self.dish.post(signal=self.food_tally, period=self.tally_cycle)

        # plots
        self.plots = {"total_cells": 0, "stem_cells": 0, "terminal_cells": 0, "base": 0}

        balls = self.dish.get_agents(condition=lambda a: isinstance(a, Ball))
        min_ball_radius = 0.006 # min([ball.radius for ball in balls])
        max_ball_radius = 0.010 # max([ball.radius for ball in balls])

        stride = self.dish.get_upper_bounds() - self.dish.get_bottom_bounds()
        width, height = 2000, 2000
        windows_scale, simulation_scale = (width + height) / 2, (stride[0] + stride[1]) / 2

        min_radius = int(20 * windows_scale * (min_ball_radius / simulation_scale))
        max_radius = int(20 * windows_scale * (max_ball_radius / simulation_scale))

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        # balls radius
        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if isinstance(agent, Ball) else EPSILON)

        # init base class
        super().__init__(universe=self.dish, agents_filter=lambda a: isinstance(a, Cellulata),
                         width=width, height=height, min_radius=min_radius/10, max_radius=max_radius/10, plotter_period=1550)

    def get_plots(self):
        return self.plots

    def setup_case_two_with_chalones_varying_amplification(self):
        # cells
        for i in range(0, 40):
            # initial conditions
            x, y = random_state.uniform(0.2, 0.8), random_state.uniform(0.2, 0.8)

            """ create gene pool (varying amplification) """
            gene_varying_amplification_pool[Trait.AMPLIFICATION] = (int(GENE_CAPACITY / 3), int(2 * GENE_CAPACITY / 3))
            gene_pool = GenePool(capacity=GENE_CAPACITY, genes=gene_varying_amplification_pool)
            gene = gene_pool.get_random_gene(random_state)

            # instantiate bug
            cell = Stem(gene=gene, dish=self.dish, position=numpy.array([x, y]), signal=True)
            self.dish.put(cell)

    def injury(self, agent_filter=lambda agent: agent, loss=0.80):
        # cell index
        index = [list(range(2, self.dish.get_grid() - 2))]

        # total of cells
        depleted = int(loss * (self.dish.get_grid() - 4) ** 2)
        for n in range(1, self.dish.get_dimensions()):
            index.append(list(range(2, self.dish.get_grid() - 2)))

        # put it
        for i, idx in enumerate(itertools.product(*index)):
            stride = (self.dish.get_upper_bounds() - self.dish.get_bottom_bounds()) / self.dish.get_grid()
            position = stride / 2 + numpy.array(idx) * stride

            cell = self.dish.get_cell(position)
            for each in filter(agent_filter, cell.get_children()):
                each.apoptosis = True

            cell.chalones = 0

            # deplete loss fraction
            if (i + 1) == depleted:
                break

    def on_mouse_release(self, x, y, button):
        cell = self.dish.get_cell([x, y])
        for each in cell.get_children():
            each.apoptosis = True

    def done(self):
        return False

    def step(self):
        toc = time.process_time()

        # warps in food and bugs
        stem_cells, terminal_cells = 0, 0
        for agent in self.dish.get_agents():
            if isinstance(agent, Stem):
                stem_cells += 1
            elif isinstance(agent, Terminal):
                terminal_cells += 1

        # count cells
        total_cells = stem_cells + terminal_cells
        self.plots["total_cells"], self.plots["base"] = 0, 0
        self.plots["stem_cells"] = stem_cells / Tissue.CELL_SCALE
        self.plots["terminal_cells"] = terminal_cells / Tissue.CELL_SCALE

        if self.dish.get_time() % self.tally_cycle == 0:
            # print some stats
            print("[.] ---- time          :", self.dish.get_time())
            print(f"[-] cost (average)     : {self.dish.get_average_cost():.2f}")
            # warps in food and bugs
            bug_warps, food_warps = 0, 0
            for agent in self.dish.get_agents():
                if isinstance(agent, Cellulata):
                    bug_warps += agent.warps
                elif isinstance(agent, Food):
                    food_warps += agent.warps

            # total warps
            colony_warps = self.dish.warps + bug_warps + food_warps
            print("[w] total cells        :", total_cells)
            print("[w] stem cells         :", stem_cells)
            print("[w] terminal cells     :", terminal_cells)
            print(f"[w] dish warps         : {self.dish.warps:.2f} ({self.dish.warps / colony_warps:.2f})")
            print(f"[w] bug warps          : {bug_warps:.2f} ({bug_warps / colony_warps:.2f})")
            print(f"[w] food warps         : {food_warps:.2f} ({food_warps / colony_warps:.2f})")
            print(f"[w] colony warps       : {colony_warps:.2f} ({colony_warps / colony_warps:.2f})")

            # get the top genes
            top_genes = {gene: count for gene, count in sorted(self.collect_tally.genes.items(),
                                                               key=lambda item: item[1]["count"], reverse=True)}

            # print top genes
            for i, gene in enumerate(top_genes):
                print("[@]", " {:04}".format(top_genes[gene]["count"]), gene,
                      "E = {:03.2f}".format(float(top_genes[gene]["warps"]) / float(top_genes[gene]["count"])))
                if i == Tissue.TOP_GENES:
                    break
