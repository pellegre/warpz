from warpz.simulation.local import *
from models.machines.media import *
from models.plotter.window import *

# bug radius
MAX_RADIUS = 30
MIN_RADIUS = 5

# grid size
GRID = 10

# initial glucose seed
WARPS_SEED = 1400 * GRID * GRID


def stem_machine():
    stem = Machine(name="mother")

    stem.add_state("stem")
    stem.add_state("renew", flux=[Flow("grow")])
    stem.add_state("differentiate", flux=[Flow("chalone")])
    stem.add_state("cell", flux=Flow("chalone"))

    stem.add_flow("stem", "renew", flow=Flow("grow"))

    stem.add_flow("renew", "stem", flow=Flow("chalone"))

    stem.add_flow("renew", "differentiate", flow=Flow("grow"))

    stem.add_flow("differentiate", "cell", flow=Flow("chalone"))

    return stem


class Tissue(Simulation):
    TOP_GENES = 5

    def __init__(self, knock_genes=None):
        # display cycle
        self.display_cycle = 40

        # machines
        self.stem_machine = stem_machine()

        # chromo
        self.chromo = {"stem": 0.10, "renew": 0.30, "differentiate": 0.70}

        # medium
        self.medium = Medium(upper_bounds=[1.50, 0.10], bottom_bounds=[0.0, 0.0], boundary=Box.Boundary.CLOSED,
                             tracking_type=Box.Tracking.QUAD, warps=WARPS_SEED,
                             machines=[self.stem_machine], chromo=self.chromo)

        self.instance_tissue()

        balls = self.medium.get_agents(condition=lambda a: isinstance(a, Ball))
        min_ball_radius = min([ball.radius for ball in balls])
        max_ball_radius = max([ball.radius for ball in balls])

        stride = self.medium.get_upper_bounds() - self.medium.get_bottom_bounds()
        width, height = 2715, 1530
        windows_scale, simulation_scale = (width + height) / 2, (stride[0] + stride[1]) / 2

        min_radius = int(5 * windows_scale * (min_ball_radius / simulation_scale))
        max_radius = int(5 * windows_scale * (max_ball_radius / simulation_scale))

        print("[+] minimal ball radius", min_radius)
        print("[+] maximum ball radius", max_radius)

        # balls radius
        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if isinstance(agent, Ball) else EPSILON)

        # color property
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.color if hasattr(agent, "color") else 0)

        # time counter
        self.tic = time.process_time()

        super().__init__(universe=self.medium, agents_filter=lambda a: isinstance(a, Ball),
                         width=width, height=height, min_radius=min_radius, max_radius=max_radius)

    def instance_tissue(self):
        # get bounds
        bottom, upper = self.medium.get_bottom_bounds(), self.medium.get_upper_bounds()

        # single releasing plus rear cells
        for i in range(0, 25):
            x, y = random_state.uniform(bottom[0], 0.20), random_state.uniform(bottom[1], upper[1])
            self.medium.instantiate(position=numpy.array([x, y]), machine="mother", state="renew")

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.medium.get_mesh()

    def done(self):
        return False

    @staticmethod
    def on_mouse_press(x, y, button):
        print("[+] press", x, y)

    def step(self):
        # print some stats
        if self.medium.get_time() % self.display_cycle == 0:
            # check time
            toc = time.process_time()

            # cellulatas
            cellulatas = set(self.medium.get_agents(condition=lambda a: isinstance(a, Cellulata)))

            # chemokines (in unit cells)
            cells = self.medium.get_agents(condition=lambda a: isinstance(a, Cell))

            # glucose distribution
            free_warps = sum([cell.material[WARPS] for cell in cells])

            # total glucose
            total_glucose = self.medium.warps + free_warps

            if len(cellulatas):
                print("-------------- (step) ", self.medium.get_time())
                universe_schwifties = self.medium.get_schwifties()
                agents_schwifties = sum([a.get_schwifties() for a in self.medium.get_agents()])

                print(f"[-] schwifty         : {agents_schwifties / universe_schwifties:.2f}")
                print(f"[-] price            : {1 / self.medium.warps_price:.2f}")
                print(f"[-] time             : {toc - self.tic:.2f}")

                print("[#] --> cells        :", len(cellulatas))

                print(f"[#] --> warps        : {total_glucose:.2f}")
                print(f"[#] free             : {free_warps:.2f}")
                print(f"[#] metabolized      : {self.medium.warps:.2f}")

            # update time
            self.tic = toc
