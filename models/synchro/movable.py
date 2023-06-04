from warpz.simulation.local import *
from models.synchro.units import *
from models.gene.pool import *

import numpy
import pandas
import os

from glumpy import gl, gloo, data
from glumpy.geometry import primitives
from glumpy.transforms import Viewport, PanZoom
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

mesh_vertex = """
attribute vec3 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = <transform(vec4(position.xy,0,1.0))>;
    v_texcoord = texcoord;
}
"""

mesh_fragment = """
#include "misc/spatial-filters.frag"
#include "colormaps/colormaps.glsl"

uniform sampler2D data;
uniform vec2 data_shape;
varying vec2 v_texcoord;

void main()
{
    // Extract data value
    float value = Bicubic(data, data_shape, v_texcoord).r;
    // Map value to rgb color
    vec4 bg_color = vec4(colormap_hot(value), 1.0);
    gl_FragColor = bg_color;
} """


# signals windows
signals = app.Window(1800, 1980)
thetas = app.Window(460, 1980)

# green pointer action correlation matrix
green_correlation = app.Window(500, 500)
red_correlation = app.Window(500, 500)
cross_correlation = app.Window(500, 500)

# number of players
NUMBER_OF_PLAYERS = 12

# time scale
TIME_SCALE, TIME_FACTOR = 20000, 100

# output folder
OUTPUT_FOLDER = "./models/synchro/output/"

# plot colors
TILES = 2 * NUMBER_OF_PLAYERS + 2
COLORS = numpy.random.uniform(0.5, 1.0, (TILES, 4))
COLORS[NUMBER_OF_PLAYERS] = [0, 0, 0, 0]
COLORS[NUMBER_OF_PLAYERS + 1] = [0, 0, 0, 0]


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


class PlotMesh:
    def __init__(self, windows, shape):
        self.windows = windows

        # space program
        self.program = gloo.Program(mesh_vertex, mesh_fragment)
        self.vertices, self.indices = primitives.plane(2.0, n=64)
        self.program.bind(self.vertices)

        self.pan_zoom_transform = PanZoom(aspect=1)
        self.program['transform'] = self.pan_zoom_transform
        self.windows.attach(self.pan_zoom_transform)

        # mesh
        self.data = numpy.zeros(shape=shape)
        self.min, self.max = -1.00, 1.00

        # visitor
        self.visitor = None

    def update_background_mesh(self):
        if self.data is not None:
            self.program['data'] = (self.data - self.min) / (self.max - self.min)
            self.program['data'].interpolation = gl.GL_NEAREST
            self.program['data_shape'] = self.data.shape[1], self.data.shape[0]
            self.program['u_kernel'] = data.get("spatial-filters.npy")
            self.program['u_kernel'].interpolation = gl.GL_LINEAR

    def hook_visitor(self, visitor):
        self.visitor = visitor


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

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_mesh(self.positions)


class SignalMatrix(PlotMatrix):
    def __init__(self):
        super(SignalMatrix, self).__init__(rows=TILES, cols=1, points=TIME_SCALE)

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_signal(self.positions)


class ThetaMatrix(PlotMatrix):
    def __init__(self):
        super(ThetaMatrix, self).__init__(rows=TILES, cols=1, points=max(STATES_SIZE, NU_BINS))

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_theta(self.positions)


# signal matrix
signal_matrix = SignalMatrix()

# theta matrix
theta_matrix = ThetaMatrix()

# correlation meshes
green_correlation_mesh = PlotMesh(windows=green_correlation, shape=(NUMBER_OF_PLAYERS, NUMBER_OF_PLAYERS))
red_correlation_mesh = PlotMesh(windows=red_correlation, shape=(NUMBER_OF_PLAYERS, NUMBER_OF_PLAYERS))
cross_correlation_mesh = PlotMesh(windows=cross_correlation, shape=(NUMBER_OF_PLAYERS, NUMBER_OF_PLAYERS))

font = FontManager.get("OpenSans-Regular.ttf", 12, mode='agg')
label_signal = GlyphCollection('agg', transform=OrthographicProjection(Position()))
label_signal.append("Spikes (time)", font, origin=(300, 80, 0), color=(1, 1, 1, 1))

signals.attach(label_signal["transform"])


@signals.event
def on_draw(dt):
    signals.set_position(600, 0)
    signals.set_title("signal spikes (time)")

    signals.clear()
    signal_matrix.lines.draw()

    # visit on draw
    signal_matrix.visit_on_draw()

    signal_matrix.positions[:, 0] = signal_matrix.positions[:, 1]
    signal_matrix.positions[:, -1] = signal_matrix.positions[:, -2]


@thetas.event
def on_draw(dt):
    thetas.set_position(0, 0)
    thetas.set_title("learning")

    thetas.clear()
    theta_matrix.lines.draw()

    # visit on draw
    theta_matrix.visit_on_draw()

    # Here we ensure:
    #   * first point = second point
    #   * last point = prev last point
    theta_matrix.positions[:, 0] = theta_matrix.positions[:, 1]
    theta_matrix.positions[:, -1] = theta_matrix.positions[:, -2]


@green_correlation.event
def on_draw(dt):
    green_correlation.set_position(3400, 2400)

    green_correlation.clear()

    green_correlation.set_title("correlation (catcher)")

    green_correlation_mesh.update_background_mesh()

    if green_correlation_mesh.data is not None:
        green_correlation_mesh.program.draw(gl.GL_TRIANGLES, green_correlation_mesh.indices)


@red_correlation.event
def on_draw(dt):
    red_correlation.set_position(2827, 2400)

    red_correlation.clear()

    red_correlation.set_title("correlation (target)")

    red_correlation_mesh.update_background_mesh()

    if red_correlation_mesh.data is not None:
        red_correlation_mesh.program.draw(gl.GL_TRIANGLES, red_correlation_mesh.indices)


@cross_correlation.event
def on_draw(dt):
    cross_correlation.clear()
    cross_correlation.set_position(2320, 2400)

    cross_correlation.set_title("cross correlation")

    cross_correlation_mesh.update_background_mesh()

    if cross_correlation_mesh.data is not None:
        cross_correlation_mesh.program.draw(gl.GL_TRIANGLES, cross_correlation_mesh.indices)


# controllers
POINTER_CONTROLLER = BayesUnit
TARGET_CONTROLLER = BayesUnit

RUNNER_SPEED = 0.03
FOLLOWER_SPEED = 0.03


class Movable(Simulation):
    def __init__(self):
        # setup visual properties
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.chromo if hasattr(agent, "chromo") else 0)

        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if hasattr(agent, "radius") else 0.5)

        self.set_property(Simulation.Properties.SCALE_CHROMO, 0.50)

        # define frames
        self.pointer_frame, self.action_frame, self.belief_frame = None, None, None

        # game screen
        self.screen = Screen(grid=40, boundary=Box.Boundary.CLOSED)

        # setup fixed target
        self._setup_movable_target()
        self._prepare_run()

        # hook signal matrix
        signal_matrix.hook_visitor(self)
        theta_matrix.hook_visitor(self)

        # init base class
        super().__init__(universe=self.screen, agents_filter=lambda agent: isinstance(agent, Particle),
                         plotter_period=1550, min_radius=20)

    def _setup_movable_target(self):
        # ----------------------------------------------------------------
        #
        # instantiate pointer, target and controllers
        #
        # ----------------------------------------------------------------

        # instantiate target
        x, y = random_state.uniform(0, 1), random_state.uniform(0.30, 0.50)
        self.target = Pointer(position=numpy.array([x, y]), chromo=0.1, motility=RUNNER_SPEED)
        self.screen.put(self.target)

        # instantiate pointer
        x, y = random_state.uniform(0, 1), random_state.uniform(0, 0.15)
        self.pointer = Pointer(position=numpy.array([x, y]), target=self.target, motility=FOLLOWER_SPEED)
        self.screen.put(self.pointer)

        # add players (follower)
        self.followers = [POINTER_CONTROLLER(target=self.target, pointer=self.pointer, direction=direction)
                          for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_PLAYERS + 1))
                          if i < NUMBER_OF_PLAYERS]
        for player in self.followers:
            self.screen.add(player)

        # add runners players
        self.runners = [TARGET_CONTROLLER(target=self.pointer, pointer=self.target, direction=direction, escape=True)
                        for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_PLAYERS + 1))
                        if i < NUMBER_OF_PLAYERS]
        for player in self.runners:
            self.screen.add(player)

    def _prepare_run(self):
        # ----------------------------------------------------------------
        #
        # tally frames
        #
        # ----------------------------------------------------------------

        # get running folder
        if POINTER_CONTROLLER is BayesUnit and TARGET_CONTROLLER is BayesUnit:
            self.output_folder = get_running_environment("bayes-movable")
        elif POINTER_CONTROLLER is BayesUnit and TARGET_CONTROLLER is AutomataUnit:
            self.output_folder = get_running_environment("automata-movable")
        else:
            self.output_folder = get_running_environment("bayes-automata-movable")

        self.players_columns = ["player-" + str(i) for i in range(0, len(self.followers))] + \
                               ["runner-" + str(i) for i in range(0, len(self.runners))]

        player_theta = pandas.DataFrame([[player.theta for player in self.followers] +
                                         [player.theta for player in self.runners]], columns=self.players_columns)
        player_theta.to_csv(self.output_folder + "player_theta.csv", index=False)

        self.action_frame = pandas.DataFrame(columns=["time"] + self.players_columns)

        # prepare pointer frame
        self.pointer_columns = ["time", "pointer-x", "pointer-y", "target-x", "target-y", "distance", "touch"]
        self.pointer_frame = pandas.DataFrame(columns=self.pointer_columns)

        # prepare belief distribution frame
        self.belief_columns = ["time", "player"] + ["bin-" + str(i) for i in range(0, max(STATES_SIZE, NU_BINS))]
        self.belief_frame = pandas.DataFrame(columns=self.belief_columns)

    def on_draw_signal(self, positions):
        # actions frame
        actions = [player.action_taken for player in self.followers]
        signal_time = (self.get_time() * TIME_FACTOR) % TIME_SCALE

        if signal_time == 0:
            positions[:, 1:-1, 1] = 0

        for i, action in enumerate(actions):
            if action is Action.MOVE:
                positions[i, signal_time:signal_time + 1, 1] = 1

        actions = [player.action_taken for player in self.runners]
        for i, action in enumerate(actions):
            if action is Action.MOVE:
                positions[2 + NUMBER_OF_PLAYERS + i, signal_time:signal_time + 1, 1] = 1

    def on_draw_theta(self, positions):
        for stride, players in enumerate([self.followers, self.runners]):
            reference_player = players[0]

            # plot belief
            if isinstance(reference_player, BayesUnit):
                belief_per_player = {i: player.belief.density for i, player in enumerate(players)}

                for player in belief_per_player:
                    density = belief_per_player[player]
                    for i in range(0, NU_BINS):
                        positions[stride + stride * NUMBER_OF_PLAYERS + player, i:i + 1, 1] = density[i] / max(density)
            else:
                matrix_per_player = {i: player.q_matrix[:, int(Action.MOVE)] for i, player in enumerate(players)}

                for player in matrix_per_player:
                    density = matrix_per_player[player]
                    factor = int(NU_BINS / STATES_SIZE)
                    for i in range(0, STATES_SIZE):
                        idx = i * factor
                        positions[2*stride + stride * NUMBER_OF_PLAYERS + player, idx:idx + factor, 1] = \
                            density[i] / max(density)

    @staticmethod
    def get_action_correlation(one: pandas.DataFrame, other: pandas.DataFrame):
        rows = other.columns
        columns = one.columns

        matrix = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)
        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                one_data, other_data = one[col].tail(100).values.astype(float), \
                                       other[row].tail(100).values.astype(float)

                n = len(one_data)

                z = numpy.sum((one_data - numpy.average(one_data)) * (other_data - numpy.average(other_data)))
                x = (n - 1) * numpy.std(one_data) * numpy.std(other_data)
                matrix[i, j] = z / x

        return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                                columns=[i + 1 for i in range(0, len(columns))])

    @staticmethod
    def get_correlation(matrix):
        corr = (numpy.linalg.norm(matrix.flatten()) ** 2) / (matrix.shape[0] * matrix.shape[1])
        return corr

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.screen.get_mesh()

    def done(self):
        return False

    def on_mouse_release(self, x, y, button):
        for each in self.followers:
            if isinstance(each, BayesUnit):
                each.belief = BeliefDistribution(bins=NU_BINS)
            else:
                each.q_matrix = numpy.zeros((2 * STATES_SIZE, 2))

        self.target._position = [x, y]

    def step(self):
        # ----------------------------------------------------------------
        #
        # collect data
        #
        # ----------------------------------------------------------------

        if self.get_time() > 1:
            # pointer and target position frame
            pointer_x, pointer_y = self.pointer.get_position()
            target_x, target_y = self.target.get_position()

            distance = numpy.linalg.norm(self.target.get_position() - self.pointer.get_position())

            pointer_position = [self.get_time() - 1, pointer_x, pointer_y, target_x, target_y, distance,
                                self.pointer.touch]
            pointer_frame = pandas.DataFrame([pointer_position], columns=self.pointer_columns)

            self.pointer_frame = pandas.concat([self.pointer_frame, pointer_frame])

        # actions frame (taken at each time step per player)
        actions = [player.action_taken for player in self.followers] + \
                  [player.action_taken for player in self.runners]

        action_frame = pandas.DataFrame([[self.get_time()] + actions], columns=["time"] + self.players_columns)
        self.action_frame = pandas.concat([self.action_frame, action_frame])

        # belief distribution frame
        reference_follower, reference_runner = self.followers[0], self.runners[0]
        if isinstance(reference_follower, BayesUnit):
            belief_per_player = {"player-" + str(i): player.belief.density for i, player in enumerate(self.followers)}
        else:
            belief_per_player = {"player-" + str(i): player.q_matrix[:, int(Action.MOVE)]
                                 for i, player in enumerate(self.followers)}

        if isinstance(reference_runner, BayesUnit):
            belief_per_runner = {"runner-" + str(i): player.belief.density for i, player in enumerate(self.runners)}
        else:
            belief_per_runner = {"runner-" + str(i): player.q_matrix[:, int(Action.MOVE)]
                                 for i, player in enumerate(self.runners)}

        belief_both = {**belief_per_player, **belief_per_runner}
        belief_matrix = [[self.get_time(), player] + list(belief_both[player]) +
                         [0]*(NU_BINS - len(belief_both[player])) for player in belief_both]

        belief_frame = pandas.DataFrame(belief_matrix, columns=self.belief_columns)
        self.belief_frame = pandas.concat([self.belief_frame, belief_frame])

        # ----------------------------------------------------------------
        #
        # plot action correlation
        #
        # ----------------------------------------------------------------

        green = ["player-" + str(i) for i in range(0, len(self.followers))]
        red = ["runner-" + str(i) for i in range(0, len(self.runners))]

        green_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[green], self.action_frame[green]).values

        red_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[red], self.action_frame[red]).values

        cross_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[red], self.action_frame[green]).values

        mesh_windows = [green_correlation_mesh, red_correlation_mesh, cross_correlation_mesh]

        for each in mesh_windows:
            each.min = numpy.min([mesh.data.min() for mesh in mesh_windows])
            each.max = numpy.min([mesh.data.max() for mesh in mesh_windows])

        if self.screen.get_time() % 100 == 0:
            # print some stats
            print("[.] ---- time :", self.screen.get_time())
            print("[.] ---- touching :", self.pointer.touch)

        if self.screen.get_time() % 1000 == 0:
            print("[.] ---- tally at time :", self.screen.get_time())
            self.pointer_frame.to_csv(self.output_folder + "pointer_frame.csv", index=False)
            self.action_frame.to_csv(self.output_folder + "action_frame.csv", index=False)
            self.belief_frame.to_csv(self.output_folder + "belief_frame.csv", index=False)
