from models.colony.biofilm.run.storage import *
from warpz.simulation.local import *

from glumpy import app

dish = StoredDish(filename="./models/colony/run/db/run-biofilm-no-dissipation/")

# grid
GRID = dish.get_grid()

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
        self.medium = dish

        # get coconatas
        self.coconatas = self.medium.get_coconatas()

        for row in self.coconatas[1]:
            # put coco
            coco = StoredCoco(row=row)

            # put coconata
            self.medium.put(coco)

        # get balls
        balls = self.medium.get_agents(condition=lambda a: isinstance(a, StoredCoco))

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
                          lambda agent: agent.radius if isinstance(agent, StoredCoco) else EPSILON)

        # color property
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.color if hasattr(agent, "color") else 0)

        # time counter
        self.tic = time.process_time()

        # tallies
        self.pump_stats, self.reaction_stats = dict(), dict()

        super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, StoredCoco),
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

    def step(self):
        # get substrates
        cell_substrates = self.medium.get_substrates(condition=lambda a: isinstance(a, StoredBlock))

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

        if self.get_time() % 40 == 0:
            cocos = list(self.medium.get_agents(condition=lambda a: isinstance(a, StoredCoco)))
            print("[+] time", self.get_time(), ":", len(cocos))
