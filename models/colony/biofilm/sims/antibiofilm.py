from warpz.simulation.local import *
from models.colony.biofilm.species import *
from models.plotter.window import *

import numpy
import pandas

import numpy as np
from glumpy import app
from glumpy.graphics.text import FontManager
from glumpy.transforms import Position, Trackball, Viewport
from glumpy.graphics.collections import GlyphCollection
from glumpy.graphics.collections import PathCollection
from glumpy.graphics.collections import SegmentCollection


window_plot = app.Window(width=2500, height=1200, color=(1, 1, 1, 1))


@window_plot.event
def on_key_press(key, modifiers):
    if key == app.window.key.SPACE:
        reset()


def reset():
    transform.theta = 0
    transform.phi = 0
    transform.zoom = 16.5


transform = Trackball(Position())
viewport = Viewport()
labels = GlyphCollection(transform=transform, viewport=viewport)
ticks = SegmentCollection(mode="agg", transform=transform, viewport=viewport, linewidth='local', color='local')


xmin, xmax = -2.2, 2.2
ymin, ymax = -0.9, 1.0


MAX_TIME = 230
MAX_MOLES = 150

z = 0

regular = FontManager.get("OpenSans-Regular.ttf")
n = 11
scale = 0.001
for i, y in enumerate(np.linspace(xmin, xmax, n)):
    text = "%.2f" % ((i - 3) * (MAX_MOLES / 4))
    labels.append(text, regular,
                  origin = (2.25,y,z), scale = scale, direction = (1,0,0),
                  anchor_x = "left", anchor_y="center")

    text = "%.2f" % (i*(MAX_TIME / 10))
    labels.append(text, regular, origin = (y, -0.97, z),
                  scale= scale, direction = (1,0,0),
                  anchor_x = "center", anchor_y = "top")

title = "Evolucion temporal de especies químicas en el interior / exterior de la célula"
labels.append(title, regular, origin = (0, 1.1, z),
              scale= 1.7*scale, direction = (1,0,0),
              anchor_x = "center", anchor_y = "center")

title = "tiempo (pasos)"
labels.append(title, regular, origin = (0, -1.10, z),
              scale= 1.5*scale, direction = (1,0,0),
              anchor_x = "center", anchor_y = "center")


# Frame
# -------------------------------------
P0 = [(xmin,ymin,z), (xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z)]
P1 = [(xmin,ymax,z), (xmax,ymax,z), (xmax,ymin,z), (xmin,ymin,z)]
ticks.append(P0, P1, linewidth=2)

# Grids
# -------------------------------------
n = 11
P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))

P0[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P0[:,1] = ymin
P0[:,2] = z
P1[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P1[:,1] = ymax
P1[:,2] = z
ticks.append(P0, P1, linewidth=1, color=(0,0,0,.25))

P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))
P0[:,0] = xmin
P0[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P0[:,2] = z
P1[:,0] = xmax
P1[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P1[:,2] = z
ticks.append(P0, P1, linewidth=1, color=(0,0,0,.25))


# Majors
# -------------------------------------
n = 11
P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))
P0[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P0[:,1] = ymin - 0.015
P0[:,2] = z
P1[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P1[:,1] = ymin + 0.025 * (ymax-ymin)
P1[:,2] = z
ticks.append(P0, P1, linewidth=1.5)
P0[:,1] = ymax + 0.015
P1[:,1] = ymax - 0.025 * (ymax-ymin)
ticks.append(P0, P1, linewidth=1.5)

P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))
P0[:,0] = xmin - 0.015
P0[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P0[:,2] = z
P1[:,0] = xmin + 0.025 * (xmax-xmin)
P1[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P1[:,2] = z
ticks.append(P0, P1, linewidth=1.5)
P0[:,0] = xmax + 0.015
P1[:,0] = xmax - 0.025 * (xmax-xmin)
ticks.append(P0, P1, linewidth=1.5)


# Minors
# -------------------------------------
n = 111
P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))
P0[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P0[:,1] = ymin
P0[:,2] = z
P1[:,0] = np.linspace(xmin,xmax,n)[1:-1]
P1[:,1] = ymin + 0.0125 * (ymax-ymin)
P1[:,2] = z
ticks.append(P0, P1, linewidth=1)
P0[:,1] = ymax
P1[:,1] = ymax - 0.0125 * (ymax-ymin)
ticks.append(P0, P1, linewidth=1)

P0 = np.zeros((n-2,3))
P1 = np.zeros((n-2,3))
P0[:,0] = xmin
P0[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P0[:,2] = z
P1[:,0] = xmin + 0.0125 * (xmax-xmin)
P1[:,1] = np.linspace(ymin,ymax,n)[1:-1]
P1[:,2] = z
ticks.append(P0, P1, linewidth=1)
P0[:,0] = xmax
P1[:,0] = xmax - 0.0125 * (xmax-xmin)
ticks.append(P0, P1, linewidth=1)


reset()

title = "(CN)4"
labels.append(title, regular, origin = (-2.0, 0.90, z),
              scale=1.5*scale, direction=(1, 0, 0),
              anchor_x="center", anchor_y="center")

cn4_path = PathCollection(mode="agg+", transform=transform, viewport=viewport)
for i, y in enumerate(np.linspace(0.88, 0.91, 100)):
    cn4_path.append(numpy.array([[-1.85, y, 0], [-1.75, y, 0]]), color=(0, 0, 1, 1), closed=False)
    cn4_path["color"] = 0, 0, 1, 1

window_plot.attach(cn4_path["transform"])
window_plot.attach(cn4_path["viewport"])


r_path = PathCollection(mode="agg+", transform=transform, viewport=viewport)
title = "R"
labels.append(title, regular, origin = (-2.0, 0.80, z),
              scale=1.5*scale, direction=(1, 0, 0),
              anchor_x="center", anchor_y="center")

for i, y in enumerate(np.linspace(0.78, 0.81, 100)):
    r_path.append(numpy.array([[-1.85, y, 0], [-1.75, y, 0]]), color=(1, 0, 0, 1), closed=False)
    r_path["color"] = 1, 0, 0, 1

window_plot.attach(r_path["transform"])
window_plot.attach(r_path["viewport"])

atb_path = PathCollection(mode="agg+", transform=transform, viewport=viewport)

title = "ATB"
labels.append(title, regular, origin = (-2.0, 0.70, z),
              scale=1.5*scale, direction=(1, 0, 0),
              anchor_x="center", anchor_y="center")

for i, y in enumerate(np.linspace(0.68, 0.71, 100)):
    atb_path.append(numpy.array([[-1.85, y, 0], [-1.75, y, 0]]), color=(0, 0, 0, 1), closed=False)
    atb_path["color"] = 0, 0, 0, 1

window_plot.attach(atb_path["transform"])
window_plot.attach(atb_path["viewport"])


@window_plot.event
def on_draw(dt):
    window_plot.set_position(1330, 0)

    window_plot.clear()
    ticks.draw()
    labels.draw()

    r_path.draw()
    cn4_path.draw()
    atb_path.draw()


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
        pressure = ["CNCN", "CC", "CCCC"]

        super().__init__(target=target, anabolic=anabolic, catabolic=catabolic,
                         pressure=pressure, matrix=False, **kwargs)


class Splash(Specie):
    def __init__(self, universe, position, replications=0, container=None):
        # setup gene
        chromosome = SplashSpecie()

        # initial container
        if container is None:
            container = {"CNCNCNCN": CHAMBER_MOLES // 4, "CNCNCN": CHAMBER_MOLES // 4, "CNCN": CHAMBER_MOLES // 4,
                         "CN": CHAMBER_MOLES // 4, "CCC": CHAMBER_MOLES // 16, "CCCC": CHAMBER_MOLES // 16,
                         "Z": CHAMBER_MOLES // 8}

        # initialize
        super().__init__(universe=universe, position=position, mother=chromosome, container=container)

        # replications
        self.replications = replications

    # ======
    # replicate
    #
    def replicate(self):
        # get anabolic drive moles
        moles = self.get_substrate(self._chromosome.get_anabolic_drive())

        if moles > CHAMBER_MOLES:
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
                                        universe=self.get_universe()))

    # ======
    # interact
    #
    def interact(self, signals):
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

        if self.is_unbalanced() and self.get_time() > 5:
            # create pump
            pump = Gene(AntiBioReactor.ANTIBIOTIC_RIGHT, Action.OUTWARD, moles=Gene.MOLES-1)
            pump.init(self)

            # insert pump
            self.get_chromosome().insert_pump(pump)

        self.color = 0.10

    def get_radius(self):
        return 0.01

    def get_color(self):
        return 0.10

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

        # reward point
        self.reward_point = [[-2.2, -0.9, 0]]
        self.target_molecule = [[-2.2, -0.9, 0]]
        self.atb_molecule = [[-2.2, -0.9, 0]]

        super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Particle),
                         width=WIDTH, height=HEIGHT, min_radius=min_radius, max_radius=max_radius)

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.medium.get_substrate_mesh("E")

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

    def put_antibiotics(self):
        # chemokines (in unit cells)
        cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))

        for cell in cells:
            cell.put_substrate(AntiBioReactor.ANTIBIOTIC_RIGHT, WARPS_SEED // 3)
            cell.put_substrate(AntiBioReactor.ANTIBIOTIC_LEFT, WARPS_SEED // 3)

    def draw_reward(self):
        # append point
        self.reward_point.append([xmin + self.get_time() / MAX_TIME,
                                  ymin + self.splash.get_substrate("R") / MAX_MOLES, 0])

        # draw it
        r_path.append(numpy.array(self.reward_point), color=(1, 0, 0, 1), closed=False)

    def draw_target(self):
        # append point
        self.target_molecule.append([xmin + self.get_time() / MAX_TIME,
                                     ymin + self.splash.get_substrate("CNCNCNCN") / MAX_MOLES, 0])

        # draw it
        cn4_path.append(numpy.array(self.target_molecule), color=(0, 0, 1, 1), closed=False)

    def draw_atb(self):
        # append point
        self.atb_molecule.append([xmin + self.get_time() / MAX_TIME,
                                       ymin + self.splash.get_substrate(AntiBioReactor.ANTIBIOTIC_RIGHT) /
                                  MAX_MOLES, 0])

        # draw it
        atb_path.append(numpy.array(self.atb_molecule), color=(0, 0, 0, 1), closed=False)

    def step(self):
        # draw reward
        self.draw_reward()

        # draw target
        self.draw_target()

        # draw atb
        self.draw_atb()

        # get substrates
        cell_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, Block))

        # set chemical meshes
        chemical_n_window.set_mesh(self.medium.get_substrate_mesh("N"))
        chemical_z_window.set_mesh(self.medium.get_substrate_mesh("Z"))
        chemical_r_window.set_mesh(self.medium.get_substrate_mesh("R"))
        chemical_e_window.set_mesh(self.medium.get_substrate_mesh(AntiBioReactor.ANTIBIOTIC_RIGHT) // 4)

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

        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        # put atb
        if self.get_time() == 50:
            self.put_antibiotics()

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

            for i, each in enumerate(coconatas):
                print("[#] unbalanced       :", "true" if each.is_unbalanced() else "false")
                print("[#] starved          :", "true" if each.is_starved() else "false")
                print("[#] scarce           :", "true" if each.is_scarce() else "false")

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
