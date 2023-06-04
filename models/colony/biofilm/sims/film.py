from warpz.simulation.local import *

from models.colony.biofilm.species import *

from models.plotter.window import *

import numpy
import pandas

# ===========
#
# plots
#
# ===========

# grid size
GRID = (24, 10)

# chemical A matrix
chemical_c_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_n_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_z_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_r_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))
chemical_e_window = PlotMesh(windows=app.Window(750, 750), shape=(GRID[1], GRID[0]))


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

    chemical_e_window.windows.set_title("G")

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
WIDTH, HEIGHT = 3815, 1200

# upper bounds
TOP = 2.00
LENGTH = 9.00

# energy seed
WARPS_SEED = int(CHAMBER_MOLES)

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

        # nutrients
        food = ["CCCC", "CCC", "CC", "N", "C", "CN", "CNCN", "NC", "NCNC"]

        # distribute on cells
        cells = self.medium.get_cells()
        for cell in cells:
            # put nutrients
            for each in food:
                cell.put_substrate(each, 2 * WARPS_SEED)

            cell.put_substrate("N", 5 * WARPS_SEED)
            cell.put_substrate("Z", 2 * WARPS_SEED)

        for cell in self.medium.get_cells():
            # get delta
            delta = (self.medium.get_upper_bounds() - self.medium.get_bottom_bounds()) / self.medium.get_grid()

            # get cell index
            index = cell.index

            # get position
            position = self.medium.get_bottom_bounds() + index * delta + delta / 2

            # sample cell
            if random_state.uniform(0, 1) < 0.50:
                # coconata
                self.medium.put(RightCoco(universe=self.medium, position=position, matrix=True))

            else:
                # coconata
                self.medium.put(LeftCoco(universe=self.medium, position=position, matrix=True))

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

        super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Particle),
                         width=WIDTH, height=HEIGHT, min_radius=min_radius, max_radius=max_radius)

    def put_antibiotics(self):
        # chemokines (in unit cells)
        cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))

        for cell in cells:
            cell.put_substrate(AntiBioReactor.ANTIBIOTIC_RIGHT, 20 * WARPS_SEED)
            cell.put_substrate(AntiBioReactor.ANTIBIOTIC_LEFT, 20 * WARPS_SEED)

    def should_plot(self):
        return False

    def get_mesh(self):
        return numpy.log(self.medium.get_substrate_mesh("E") + 1)

    def done(self):
        return False

    @staticmethod
    def on_mouse_press(x, y, button):
        print("[+] press", x, y)

    @staticmethod
    def is_c_molecule(substrate):
        return all([c == 'C' for c in substrate])

    @staticmethod
    def is_cn_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("CN") == len(substrate) // 2

    @staticmethod
    def is_nc_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("NC") == len(substrate) // 2

    def step(self):
        # get substrates
        cell_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Block))

        # set chemical meshes
        chemical_n_window.set_mesh(self.medium.get_substrate_mesh("N"), minimum=0)
        chemical_z_window.set_mesh(self.medium.get_substrate_mesh("Z"), minimum=0)
        chemical_r_window.set_mesh(self.medium.get_substrate_mesh("R"), minimum=0)
        chemical_e_window.set_mesh(numpy.log(self.medium.get_gene_flux_mesh() + 1))

        # get cc polymers
        cc_substrates = [s for s in cell_substrates if self.is_c_molecule(s)]
        if len(cc_substrates) > 0:
            cc_mesh = self.medium.get_substrate_mesh(cc_substrates[0])
            for i in range(1, len(cc_substrates)):
                cc_mesh += self.medium.get_substrate_mesh(cc_substrates[i])

            # set chemical meshes
            chemical_c_window.set_mesh(cc_mesh, minimum=0)

        # get cn polymers
        n_substrates = [s for s in cell_substrates if self.is_cn_molecule(s) or self.is_nc_molecule(s) or s == "N"]

        if len(n_substrates) > 0:
            cn_mesh = self.medium.get_substrate_mesh(n_substrates[0])
            for i in range(1, len(n_substrates)):
                cn_mesh += self.medium.get_substrate_mesh(n_substrates[i])

            # set chemical meshes
            chemical_n_window.set_mesh(cn_mesh, minimum=0)

        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        # accumulate cell tallies
        for coco in coconatas:
            # accumulate state
            for pump in coco.get_chromosome().get_pumps():
                decision = pump.get_action()

                # pump
                each = pump.get_name()[pump.get_name().find("@") + 1:]

                if decision is not Action.NONE:
                    if each not in self.pump_stats:
                        self.pump_stats[each] = {Action.INWARD: 0, Action.OUTWARD: 0, Action.BUILD: 0, Action.BREAK: 0}

                    self.pump_stats[each][decision] += pump.get_moles()

            for reaction in coco.get_chromosome().get_reactions():
                decision = reaction.get_action()
                each = reaction.get_name()[reaction.get_name().find("@") + 1:]

                if decision is not Action.NONE:
                    if each not in self.reaction_stats:
                        self.reaction_stats[each] = {Action.POLYMERIZATION: 0, Action.CLEAVAGE: 0}

                    self.reaction_stats[each][decision] += reaction.get_moles()

        # put antibiotics
        # if self.medium.get_time() == 200:
        #     self.put_antibiotics()

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
