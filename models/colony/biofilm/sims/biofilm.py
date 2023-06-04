from warpz.simulation.local import *
from models.colony.biofilm.species import *
from models.plotter.window import *

import pandas

from glumpy import app


# ===========
#
# plots
#
# ===========

# grid size
GRID = (14, 14)

# chemical A matrix
chemical_c_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_n_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_z_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_r_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_e_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))

MAX_TIME = 100
MAX_MOLES = 5

# plot graph
graph = Graph(x_limits=(0, MAX_TIME), y_limits=(0, MAX_MOLES),
              title="Evolucion temporal de especies químicas en el interior / exterior de la célula",
              x_title="tiempo (pasos)", y_title="cantidad (moleculas)")

# add tickers
graph.add_ticker("(CN)4", color=(0, 0, 1, 1))
graph.add_ticker("R", color=(1, 0, 0, 1))
graph.add_ticker("Z", color=(0, 0, 0, 1))


@graph.window.event
def on_draw(dt):
    graph.on_draw(dt)


@chemical_c_window.windows.event
def on_draw(dt):
    chemical_c_window.windows.clear()

    chemical_c_window.windows.set_position(0, 2400)

    chemical_c_window.windows.set_title("C")

    chemical_c_window.on_draw(dt)


@chemical_n_window.windows.event
def on_draw(dt):
    chemical_n_window.windows.clear()

    chemical_n_window.windows.set_position(857, 2400)

    chemical_n_window.windows.set_title("N")

    chemical_n_window.on_draw(dt)


@chemical_z_window.windows.event
def on_draw(dt):
    chemical_z_window.windows.clear()

    chemical_z_window.windows.set_position(857 + 750, 2400)

    chemical_z_window.windows.set_title("Z")

    chemical_z_window.on_draw(dt)


@chemical_r_window.windows.event
def on_draw(dt):
    chemical_r_window.windows.clear()

    chemical_r_window.windows.set_position(857 + 2 * 750, 2400)

    chemical_r_window.windows.set_title("R")

    chemical_r_window.on_draw(dt)


@chemical_e_window.windows.event
def on_draw(dt):
    chemical_e_window.windows.clear()

    chemical_e_window.windows.set_position(857 + 3 * 750, 2400)

    chemical_e_window.windows.set_title("ATB")

    chemical_e_window.on_draw(dt)


# ===========
#
# universe parameters
#
# ===========

# bug radius
MAX_RADIUS = 14
MIN_RADIUS = 8

# windows width / height
WIDTH, HEIGHT = 1200, 1200

# upper bounds
TOP = 1.00
LENGTH = 1.00

# energy seed
WARPS_SEED = int(CHAMBER_MOLES)

# count lysis
lysis = 0

# ===========
#
# bio film
#
# ===========


class SplashSpecie(Mother):
    def __init__(self, **kwargs):
        # anabolic target molecule
        target = "CNCNCNCN"

        # reaction units
        anabolic = [("CNCN", "CNCN")]
        catabolic = [("CC", "CC"), ("C", "C")]

        # pump space
        pressure = ["CNCN", "CC", "CCCC", "CNCNCNCN"]

        super().__init__(target=target, anabolic=anabolic, catabolic=catabolic,
                         pressure=pressure, matrix=False, **kwargs)


class Splash(Specie):
    def __init__(self, universe, position, replications=0, container=None, chromosome=None):
        # setup gene
        self.chromosome = SplashSpecie() if chromosome is None else chromosome

        # initial container
        if container is None:
            container = {"CNCNCNCN": CHAMBER_MOLES // 4, "CNCN": CHAMBER_MOLES // 4,
                         "CCCC": CHAMBER_MOLES // 16, "Z": CHAMBER_MOLES // 8}

        # initialize
        super().__init__(universe=universe, position=position, mother=self.chromosome, container=container)

        # replications
        self.replications = replications

    # ======
    # replicate
    #
    def replicate(self):
        # get anabolic drive moles
        moles = self.get_substrate(self._chromosome.get_anabolic_drive())

        if moles > CHAMBER_MOLES // 2:
            # get substrates
            substrates = self.get_container()

            # craft container for new born
            born_container = dict()
            for s in substrates:
                # splash it
                born_container[s] = substrates[s] // 2

                # balance
                self.put_substrate(s, -born_container[s])

            # replicate
            yield Replicate(born=Splash(position=self.get_position(), container=born_container,
                                        universe=self.get_universe(), chromosome=copy.deepcopy(self.chromosome)))

    # ======
    # interact
    #
    def interact(self, signals):
        # global counter
        global lysis

        if self.get_moles() > LYSIS_MOLES:
            # cell lysis
            lysis += 1

        if not self.done():
            # diffuse
            self.diffuse_inward()

            # sense the environment
            self.sense()

            # catalytic reaction
            self.react_and_pump()

            # thermal reactions
            self.thermal()

            # diffuse
            self.diffuse_outward()

            # move
            yield self.move()

            # coco splash
            yield self.replicate()

        else:
            # lysis condition
            yield Gone()

    def get_radius(self):
        return 0.01

    def get_color(self):
        return self.get_color_by_state()

# ===========
#
# bio film
#
# ===========


class BioFilm(Simulation):
    def __init__(self):
        # display cycle
        self.display_cycle = 40

        # medium
        self.medium = Dish(upper_bounds=[LENGTH, TOP], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.OPEN,
                           tracking_type=Box.Tracking.QUAD, chemical=FilmChemical(warps=WARPS_SEED), grid=GRID)

        for cell in self.medium.get_cells():
            cell.put_substrate("CCCC", 10 * WARPS_SEED)
            cell.put_substrate("CC", WARPS_SEED)
            cell.put_substrate("CNCN", WARPS_SEED)
            cell.put_substrate("Z", 20 * WARPS_SEED)

        # coconata
        self.splash = Splash(universe=self.medium, position=[0.50, 0.50])
        self.medium.put(self.splash)

        # get balls
        balls = self.medium.get_agents(condition=lambda a: isinstance(a, Particle))

        if len(balls) > 0:
            min_ball_radius = min([ball.radius for ball in balls])
            max_ball_radius = max([ball.radius for ball in balls])

            stride = self.medium.get_upper_bounds() - self.medium.get_bottom_bounds()
            windows_scale, simulation_scale = (WIDTH + HEIGHT) / 2, (stride[0] + stride[1]) / 2

            min_radius = int(5 * windows_scale * (min_ball_radius / simulation_scale))
            max_radius = int(5 * windows_scale * (max_ball_radius / simulation_scale))
        else:
            min_radius, max_radius = 0.5, 1

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        # balls radius
        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if isinstance(agent, Particle) else EPSILON)

        # color property
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.color if hasattr(agent, "color") else 0)

        # time counter
        self.tic = time.process_time()

        # tallies
        self.pump_stats, self.reaction_stats = dict(), dict()

        # count critical states
        self.critical = 0

        # initialize simulation
        super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Particle),
                         width=WIDTH, height=HEIGHT, min_radius=min_radius, max_radius=max_radius)

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.medium.get_substrate_mesh("E")

    def done(self):
        coconatas = self.medium.get_agents(condition=lambda a: isinstance(a, Particle))
        return len(coconatas) == 0

    @staticmethod
    def on_mouse_press(x, y, button):
        print("[+] press", x, y)

    @staticmethod
    def is_c_molecule(substrate):
        return all([c == 'C' for c in substrate])

    @staticmethod
    def is_cn_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("CN") == len(substrate) // 2

    def draw_reward(self, coconatas):
        # coconata reward
        reward = sum([c.get_substrate("R") for c in coconatas]) / len(coconatas)
        graph.add_point("R", self.get_time(), reward)

    def draw_target(self, coconatas):
        # target molecules
        target = sum([c.get_substrate("CNCNCNCN") for c in coconatas]) / len(coconatas)
        graph.add_point("(CN)4", self.get_time(), target)

    def draw_z(self, coconatas):
        # energy
        z_token = sum([10 * c.get_substrate("Z") for c in coconatas]) / len(coconatas)
        graph.add_point("Z", self.get_time(), z_token)

    def step(self):
        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        if len(coconatas) > 0:
            # count critical states
            self.critical = 0

            # draw reward
            self.draw_reward(coconatas)

            # draw target
            self.draw_target(coconatas)

            # draw z
            self.draw_z(coconatas)

            # get substrates
            cell_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Block))

            # set chemical meshes
            chemical_n_window.set_mesh(self.medium.get_substrate_mesh("N"))
            chemical_z_window.set_mesh(self.medium.get_substrate_mesh("Z"))
            chemical_r_window.set_mesh(self.medium.get_substrate_mesh("R"))
            chemical_e_window.set_mesh(self.medium.get_gene_flux_mesh())

            # get cc polymers
            cc_substrates = [s for s in cell_substrates if self.is_c_molecule(s)]
            if len(cc_substrates) > 0:
                cc_mesh = self.medium.get_substrate_mesh(cc_substrates[0])
                for i in range(1, len(cc_substrates)):
                    cc_mesh += self.medium.get_substrate_mesh(cc_substrates[i])

                # set chemical meshes
                chemical_c_window.set_mesh(cc_mesh)

            # get cn polymers
            n_substrates = [s for s in cell_substrates if "N" in s]

            if len(n_substrates) > 0:
                cn_mesh = self.medium.get_substrate_mesh(n_substrates[0])
                for i in range(1, len(n_substrates)):
                    cn_mesh += self.medium.get_substrate_mesh(n_substrates[i])

                # set chemical meshes
                chemical_n_window.set_mesh(cn_mesh)

            # accumulate cell tallies
            for coco in coconatas:
                # count lysis
                if coco.is_critical():
                    self.critical += 1

                # accumulate state
                for pump in coco.get_chromosome().get_pumps():
                    decision = pump.get_action()

                    # pump
                    each = pump.get_name()[pump.get_name().find("@") + 1:]

                    if decision is not Action.NONE:
                        if each not in self.pump_stats:
                            self.pump_stats[each] = {Action.INWARD: 0, Action.OUTWARD: 0, Action.BUILD: 0, Action.BREAK: 0}

                        self.pump_stats[each][decision] += 1

                for reaction in coco.get_chromosome().get_reactions():
                    decision = reaction.get_action()
                    each = reaction.get_name()[reaction.get_name().find("@") + 1:]

                    if decision is not Action.NONE:
                        if each not in self.reaction_stats:
                            self.reaction_stats[each] = {Action.POLYMERIZATION: 0, Action.CLEAVAGE: 0}

                        self.reaction_stats[each][decision] += 1

        # print some stats
        if self.medium.get_time() % self.display_cycle == 0 or self.medium.get_time() == 1:
            # check time
            toc = time.process_time()

            # chemokines (in unit cells)
            cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))

            # get token
            warp = self.medium.get_chemistry().get_token()

            # warps in cell
            cell_warps = sum([cell.get_substrate(warp) for cell in cells])

            # warps in the dish
            dish_warps = self.medium.get_substrate(warp)

            # coconatas warps
            coconata_warps = sum([coconata.get_substrate(warp) for coconata in
                                  self.medium.get_agents(condition=lambda a: isinstance(a, Coconata))])

            # total warps
            total_warps = dish_warps + cell_warps + coconata_warps

            print("-------------- (step) ", self.medium.get_time())
            universe_schwifties = self.medium.get_schwifties()
            agents_schwifties = sum([a.get_schwifties() for a in self.medium.get_agents()])

            print(f"[-] schwifty         : {agents_schwifties / universe_schwifties:.2f}")
            print(f"[-] time             : {toc - self.tic:.2f}")

            print("[-] coconatas        :", len(coconatas))

            print(f"[#] --> warps  ({total_warps:.2f})")
            print(f"[#] dish             : {dish_warps:.2f}")
            print(f"[#] cells            : {cell_warps:.2f}")
            print(f"[#] coconatas        : {coconata_warps:.2f}")

            # dissipated energy
            dissipated_energy = self.medium.get_dissipated_energy()

            # cell energy (bonds and free)
            cell_free_energy = self.medium.get_free_energy(condition=lambda a: isinstance(a, Block))
            cell_bond_energy = self.medium.get_bond_energy(condition=lambda a: isinstance(a, Block))

            # coconata energy (bonds and free)
            coconata_free_energy = self.medium.get_free_energy(condition=lambda a: isinstance(a, Coconata))
            coconata_bond_energy = self.medium.get_bond_energy(condition=lambda a: isinstance(a, Coconata))

            # total energy
            total_energy = cell_free_energy + cell_bond_energy + coconata_bond_energy + \
                           coconata_free_energy + dissipated_energy

            # print stats
            print(f"[#] --> energy ({total_energy:.2f})")
            print(f"[#] dissipated       : {dissipated_energy:.2f}")
            print(f"[#] bond (cell)      : {cell_bond_energy:.2f}")
            print(f"[#] free (cell)      : {cell_free_energy:.2f}")
            print(f"[#] bond (coconata)  : {coconata_bond_energy:.2f}")
            print(f"[#] free (coconata)  : {coconata_free_energy:.2f}")

            moles = [c.get_moles() for c in coconatas]
            print(f"[#] average moles    : {sum(moles) / len(moles):.2f}")
            print(f"[#] average warps    : {coconata_warps / len(coconatas):.2f}")

            cell_elements = self.medium.get_elements(condition=lambda a: isinstance(a, Block))

            print("[#] --> substrates (cell)   ", {s: cell_substrates[s] for i, s in enumerate(sorted(
                cell_substrates, key=lambda r: cell_substrates[r], reverse=True))})

            print("[#] --> elements   (cell)   ", cell_elements)

            coconata_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Coconata))
            coconata_elements = self.medium.get_elements(condition=lambda a: isinstance(a, Coconata))

            print("[#] --> substrates (coco)   ", {s: round(coconata_substrates[s] / len(coconatas), 2)
                                                   for i, s in enumerate(sorted(coconata_substrates,
                                                                                key=lambda r: coconata_substrates[r],
                                                                                reverse=True))})

            print("[#] --> elements   (coco)   ", {e: round(coconata_elements[e] / len(coconatas), 2)
                                                   for e in coconata_elements})
            print("[#] --> elements   (total)  ", {e: coconata_elements[e] + cell_elements[e]
                                                   for e in coconata_elements})

            print("[#] replications     :", coconatas[0].replications)
            print("[#] lysis            :", lysis)
            print("[#] critical         :", self.critical)

            # get frames
            if len(self.reaction_stats) > 0:
                reactions = pandas.DataFrame(self.reaction_stats)
                reactions.index = ["POLY", "BREAK"]

                print("[+] reactions")
                print(reactions)

            if len(self.pump_stats) > 0:
                pumps = pandas.DataFrame({p: self.pump_stats[p] for p in self.pump_stats if p in coconata_substrates})
                pumps.index = ["INWARD", "OUTWARD", "BUILD", "BREAK"]

                # print pumps
                print("[+] pumps")
                print(pumps)

            # update time
            self.tic = toc
