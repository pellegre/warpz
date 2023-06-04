from warpz.simulation.local import *
from models.synchro.units import *
from models.gene.pool import *

import numpy
import pandas
import os

from glumpy import app, collections
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection

vertex = """
// Externs
// ------------------------------------
// extern vec3  position;
// extern float id;
// extern vec4  color;
// ... user-defined through collection init dtypes
// -----------------------------------------------
uniform float rows, cols;
varying float v_x;
varying vec4 v_color;
void main()
{
    // This line is mandatory and is responsible for fetching uniforms
    // from the underlying uniform texture
    fetch_uniforms();
    // color can end up being an attribute or a varying
    // If you want to make sure to pass it to the fragment,
    // It's better to define it here explicitly
    if (selected > 0.0)
        v_color = vec4(1,1,1,1*id);
    else
        v_color = vec4(color.rgb, color.a*id);
    float index = collection_index;
    // Compute row/col from collection_index
    float col = mod(index,cols) + 0.5;
    float row = floor(index/cols) + 0.5;
    float x = -1.0 + col * (2.0/cols);
    float y = -1.0 + row * (2.0/rows);
    float width = 0.95 / (1.0*cols);
    float height = 0.95 / (1.0*rows) * amplitude;
    v_x = xscale*position.x;
    gl_Position = vec4(x + width*xscale*position.x, y + height*position.y, 0.0, 1.0);
}
"""

fragment = """
// Collection varyings are not propagated to the fragment shader
// -------------------------------------------------------------
varying float v_x;
varying vec4 v_color;
void main(void)
{
    if( v_x < -0.95) discard;
    if( v_x > +0.95) discard;
    gl_FragColor = v_color;
}
"""

# signals windows
signals = app.Window(1800, 1920)
thetas = app.Window(460, 1920)

# number of players
NUMBER_OF_PLAYERS = 8

# time scale
TIME_SCALE, TIME_FACTOR = 20000, 100

# output folder
OUTPUT_FOLDER = "./models/synchro/output/"

# plot colors
COLORS = numpy.random.uniform(0.5, 1.0, (NUMBER_OF_PLAYERS, 4))


def get_running_environment(folder):
    # read current run ID
    run_file = open(OUTPUT_FOLDER + "/.run", "r")
    run_id = int(run_file.readline()) + 1

    print("[+] got run ID", run_id)

    if not os.path.exists(OUTPUT_FOLDER + folder):
        os.mkdir(OUTPUT_FOLDER + folder)

    run_folder = OUTPUT_FOLDER + folder + "/run-" + str(run_id) + "/"
    os.mkdir(run_folder)

    # write new run ID
    run_file = open(OUTPUT_FOLDER + "/.run", "w")
    run_file.write(str(run_id))

    return run_folder


class PlotMatrix:
    def __init__(self, rows, cols, points):
        self.rows, self.cols = rows, cols
        self.n, self.points = rows * cols, points

        # signal lines
        self.lines = collections.RawPathCollection(
            user_dtype=[("amplitude", (numpy.float32, 1), 'shared', 1),
                        ("selected", (numpy.float32, 1), 'shared', 0),
                        ("xscale", (numpy.float32, 1), 'shared', 1)],
            color="shared", vertex=vertex, fragment=fragment)

        self.lines.append(numpy.zeros((self.n * self.points, 3)), itemsize=self.points)

        self.lines["rows"] = self.rows
        self.lines["cols"] = self.cols
        self.lines["amplitude"][:self.n] = 1.0
        self.lines["color"][:self.n] = COLORS
        self.lines["color"][:self.n, 3] = 1.0
        self.lines["selected"] = 0.0
        self.lines["xscale"][:self.n] = 1

        # Each segment has two extra points for breaking the line strip
        self.positions = self.lines["position"].reshape(rows * cols, self.points + 2, 3)
        self.positions[:, 1:-1, 0] = numpy.tile(numpy.linspace(-1, 1, self.points),
                                                self.n).reshape(rows * cols, self.points)

        # Here we ensure:
        #   * first point = second point
        #   * last point = prev last point
        self.positions[:, 0] = self.positions[:, 1]
        self.positions[:, -1] = self.positions[:, -2]

        # visitor
        self.visitor = None

    def hook_visitor(self, visitor):
        self.visitor = visitor


class SignalMatrix(PlotMatrix):
    def __init__(self):
        super(SignalMatrix, self).__init__(rows=NUMBER_OF_PLAYERS, cols=1, points=TIME_SCALE)

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_signal(self.positions)


class ThetaMatrix(PlotMatrix):
    def __init__(self):
        super(ThetaMatrix, self).__init__(rows=NUMBER_OF_PLAYERS, cols=1, points=NU_BINS)

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_theta(self.positions)


# signal matrix
signal_matrix = SignalMatrix()

# theta matrix
theta_matrix = ThetaMatrix()


font = FontManager.get("OpenSans-Regular.ttf", 12, mode='agg')
label_signal = GlyphCollection('agg', transform=OrthographicProjection(Position()))
label_signal.append("Spikes (time)", font, origin=(300, 80, 0), color=(1, 1, 1, 1))

signals.attach(label_signal["transform"])


@signals.event
def on_draw(dt):
    signals.clear()
    label_signal.draw()

    signal_matrix.lines.draw()

    # visit on draw
    signal_matrix.visit_on_draw()

    signal_matrix.positions[:, 0] = signal_matrix.positions[:, 1]
    signal_matrix.positions[:, -1] = signal_matrix.positions[:, -2]


@thetas.event
def on_draw(dt):
    thetas.clear()

    theta_matrix.lines.draw()

    # visit on draw
    theta_matrix.visit_on_draw()

    # Here we ensure:
    #   * first point = second point
    #   * last point = prev last point
    theta_matrix.positions[:, 0] = theta_matrix.positions[:, 1]
    theta_matrix.positions[:, -1] = theta_matrix.positions[:, -2]


class FixedTarget(Simulation):
    def __init__(self):
        # setup visual properties
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.chromo if hasattr(agent, "chromo") else 0)

        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if hasattr(agent, "radius") else 0.5)

        self.set_property(Simulation.Properties.SCALE_CHROMO, 0.50)

        # game screen
        self.screen = Screen(grid=40, boundary=Box.Boundary.CLOSED)

        # ----------------------------------------------------------------
        #
        # instantiate pointer and target
        #
        # ----------------------------------------------------------------

        x, y = random_state.uniform(0, 1), random_state.uniform(0.70, 1)
        self.target = Pointer(position=numpy.array([x, y]), chromo=0.1, motility=0.02)
        self.screen.put(self.target)

        # instantiate pointer
        x, y = random_state.uniform(0, 1), random_state.uniform(0, 0.15)
        print("[+] initial pointer position", x, y)
        self.pointer = Pointer(position=numpy.array([x, y]), target=self.target, motility=0.02)
        self.screen.put(self.pointer)

        # add players (follower)
        self.followers = [AutomataUnit(target=self.target, pointer=self.pointer, direction=direction)
                          for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_PLAYERS + 1))
                          if i < NUMBER_OF_PLAYERS]
        for player in self.followers:
            self.screen.add(player)

        # hook visitor to the matrix
        signal_matrix.hook_visitor(self)
        theta_matrix.hook_visitor(self)

        # init base class
        super().__init__(universe=self.screen, agents_filter=lambda agent: isinstance(agent, Particle),
                         plotter_period=1550, min_radius=20)

    def on_draw_signal(self, positions):
        # actions frame
        actions = [player.action_taken for player in self.followers]
        signal_time = (self.get_time() * TIME_FACTOR) % TIME_SCALE

        if signal_time == 0:
            positions[:, 1:-1, 1] = 0

        for i, action in enumerate(actions):
            if action is Action.MOVE:
                positions[i, signal_time:signal_time + 1, 1] = 1

    def on_draw_theta(self, positions):
        reference_player = self.followers[0]

        # plot belief
        if isinstance(reference_player, BayesUnit):
            belief_per_player = {i: player.belief.density for i, player in enumerate(self.followers)}

            for player in belief_per_player:
                density = belief_per_player[player]
                for i in range(0, NU_BINS):
                    positions[player, i:i + 1, 1] = density[i] / max(density)
        else:
            matrix_per_player = {i: player.q_matrix[:, int(Action.MOVE)] for i, player in enumerate(self.followers)}

            for player in matrix_per_player:
                density = matrix_per_player[player]
                factor = int(NU_BINS / STATES_SIZE)
                for i in range(0, STATES_SIZE):
                    idx = i * factor
                    positions[player, idx:idx + factor, 1] = density[i] / max(density)

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.screen.get_mesh()

    def on_mouse_release(self, x, y, button):
        for each in self.followers:
            if isinstance(each, BayesUnit):
                each.belief = BeliefDistribution(bins=NU_BINS)
            else:
                each.q_matrix = numpy.zeros((2 * STATES_SIZE, 2))

        self.target._position = [x, y]

    def done(self):
        return False

    def step(self):
        if self.screen.get_time() % 100 == 0:
            # print some stats
            print("[.] ---- time :", self.screen.get_time())

