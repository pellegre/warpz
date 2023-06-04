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

# number of cells
NUMBER_OF_UNITS = 8

# time scale
TIME_SCALE, TIME_FACTOR = 20000, 100

# output folder
OUTPUT_FOLDER = "./models/synchro/output/"

# plot colors
TILES = 2 * NUMBER_OF_UNITS + 2
COLORS = numpy.random.uniform(0.5, 1.0, (TILES, 4))
COLORS[NUMBER_OF_UNITS] = [0, 0, 0, 0]
COLORS[NUMBER_OF_UNITS + 1] = [0, 0, 0, 0]


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

        # visitor
        self.visitor = None

    def update_background_mesh(self):
        if self.data is not None:
            self.program['data'] = (self.data - self.data.min()) / (self.data.max() - self.data.min())
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
        super(ThetaMatrix, self).__init__(rows=TILES, cols=1, points=STATES_SIZE)

    def visit_on_draw(self):
        if self.visitor is not None:
            self.visitor.on_draw_theta(self.positions)


# signal matrix
signal_matrix = SignalMatrix()

# theta matrix
theta_matrix = ThetaMatrix()

# correlation meshes
green_correlation_mesh = PlotMesh(windows=green_correlation, shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))
red_correlation_mesh = PlotMesh(windows=red_correlation, shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))
cross_correlation_mesh = PlotMesh(windows=cross_correlation, shape=(NUMBER_OF_UNITS, NUMBER_OF_UNITS))

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

    green_correlation.set_title("correlation (green)")

    green_correlation_mesh.update_background_mesh()

    if green_correlation_mesh.data is not None:
        green_correlation_mesh.program.draw(gl.GL_TRIANGLES, green_correlation_mesh.indices)


@red_correlation.event
def on_draw(dt):
    red_correlation.set_position(2827, 2400)

    red_correlation.clear()

    red_correlation.set_title("correlation (red)")

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


LEADER_SPEED = 0.005
FOLLOWER_SPEED = 0.01

LEADERS_NUMBER = 20
FOLLOWERS_NUMBER = 50


class Migration(Simulation):
    def __init__(self):
        # define frames
        self.pointer_frame, self.action_frame, self.belief_frame = None, None, None

        # game screen
        self.screen = Screen(grid=40, boundary=Box.Boundary.CLOSED)

        # setup fixed target
        self._setup_movable_target()
        self._prepare_run()

        for i in range(0, FOLLOWERS_NUMBER):
            self._create_follower_unit()

        for i in range(0, LEADERS_NUMBER):
            self._create_leader_unit()

        # hook signal matrix
        signal_matrix.hook_visitor(self)
        theta_matrix.hook_visitor(self)

        # setup visual properties
        self.set_property(Simulation.Properties.CHROMO,
                          lambda agent: agent.chromo if hasattr(agent, "chromo") else 0)

        self.set_property(Simulation.Properties.RADIUS,
                          lambda agent: agent.radius if hasattr(agent, "radius") else 0.5)

        self.set_property(Simulation.Properties.SCALE_CHROMO, 0.50)

        # init base class
        super().__init__(universe=self.screen, agents_filter=lambda agent: isinstance(agent, Particle),
                         plotter_period=1550, min_radius=6, max_radius=14)

    def _create_leader_unit(self):
        # instantiate pointer
        x, y = random_state.uniform(0, 0.50), random_state.uniform(0, 0.50)
        leader = CellulaLeader(position=numpy.array([x, y]), chromo=0.1, motility=LEADER_SPEED)
        self.screen.put(leader)

        # add cells (follower)
        leaders = [CellulataUnit(pointer=leader, direction=direction)
                   for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_UNITS + 1))
                   if i < NUMBER_OF_UNITS]
        for cell in leaders:
            self.screen.add(cell)

    def _create_follower_unit(self):
        # instantiate pointer
        x, y = random_state.uniform(0, 0.50), random_state.uniform(0, 0.50)
        follower = CellulaFollower(position=numpy.array([x, y]), motility=FOLLOWER_SPEED)
        self.screen.put(follower)

        # add cells (follower)
        followers = [CellulataUnit(pointer=follower, direction=direction)
                     for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_UNITS + 1))
                     if i < NUMBER_OF_UNITS]
        for cell in followers:
            self.screen.add(cell)

    def _setup_movable_target(self):
        # ----------------------------------------------------------------
        #
        # instantiate pointer, target and controllers
        #
        # ----------------------------------------------------------------

        # instantiate target
        x, y = random_state.uniform(0, 0.50), random_state.uniform(0, 0.50)
        self.leader = CellulaLeader(position=numpy.array([x, y]), chromo=0.1, motility=LEADER_SPEED)
        self.screen.put(self.leader)

        # instantiate pointer
        x, y = random_state.uniform(0, 0.50), random_state.uniform(0, 0.50)
        self.follower = CellulaFollower(position=numpy.array([x, y]), motility=FOLLOWER_SPEED)
        self.screen.put(self.follower)

        # add cells (follower)
        self.followers = [CellulataUnit(pointer=self.follower, direction=direction)
                          for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_UNITS + 1))
                          if i < NUMBER_OF_UNITS]
        for cell in self.followers:
            self.screen.add(cell)

        # add runners cells
        self.leaders = [CellulataUnit(pointer=self.leader, direction=direction)
                        for i, direction in enumerate(numpy.linspace(0, 2.0 * math.pi, NUMBER_OF_UNITS + 1))
                        if i < NUMBER_OF_UNITS]
        for cell in self.leaders:
            self.screen.add(cell)

    def _prepare_run(self):
        # ----------------------------------------------------------------
        #
        # tally frames
        #
        # ----------------------------------------------------------------

        # get running folder
        self.output_folder = get_running_environment("cell-migration")

        self.cells_columns = ["cell-" + str(i) for i in range(0, len(self.followers))] + \
                               ["runner-" + str(i) for i in range(0, len(self.leaders))]

        cell_theta = pandas.DataFrame([[cell.theta for cell in self.followers] +
                                         [cell.theta for cell in self.leaders]], columns=self.cells_columns)
        cell_theta.to_csv(self.output_folder + "cell_theta.csv", index=False)

        self.action_frame = pandas.DataFrame(columns=["time"] + self.cells_columns)

        # prepare pointer frame
        self.pointer_columns = ["time", "pointer-x", "pointer-y", "target-x", "target-y", "distance", "touch"]
        self.pointer_frame = pandas.DataFrame(columns=self.pointer_columns)

    def on_draw_signal(self, positions):
        # actions frame
        actions = [cell.action_taken for cell in self.followers]
        signal_time = (self.get_time() * TIME_FACTOR) % TIME_SCALE

        if signal_time == 0:
            positions[:, 1:-1, 1] = 0

        for i, action in enumerate(actions):
            if action is Action.MOVE:
                positions[i, signal_time:signal_time + 1, 1] = 1

        actions = [cell.action_taken for cell in self.leaders]
        for i, action in enumerate(actions):
            if action is Action.MOVE:
                positions[2 + NUMBER_OF_UNITS + i, signal_time:signal_time + 1, 1] = 1

    def on_draw_theta(self, positions):
        for stride, cells in enumerate([self.followers, self.leaders]):
            matrix_per_cell = {i: cell.q_matrix[:, int(Action.MOVE)] for i, cell in enumerate(cells)}

            for cell in matrix_per_cell:
                density = matrix_per_cell[cell]
                for i in range(0, STATES_SIZE):
                    positions[2*stride + stride * NUMBER_OF_UNITS + cell, i:i + 1, 1] = \
                        density[i] / max(density)

    @staticmethod
    def get_action_correlation(one: pandas.DataFrame, other: pandas.DataFrame):
        rows = other.columns
        columns = one.columns

        matrix = numpy.zeros(shape=(len(rows), len(columns)), dtype=float)
        for i, row in enumerate(rows):
            for j, col in enumerate(columns):
                one_data, other_data = one[col].values.astype(float), other[row].values.astype(float)
                n = len(one_data)

                z = numpy.sum((one_data - numpy.average(one_data)) * (other_data - numpy.average(other_data)))
                x = (n - 1) * numpy.std(one_data) * numpy.std(other_data)
                matrix[i, j] = z / x

        matrix /= numpy.sum(matrix[:, :])
        return pandas.DataFrame(matrix, index=[i + 1 for i in range(0, len(columns))],
                                columns=[i + 1 for i in range(0, len(columns))])

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.screen.get_mesh()

    def done(self):
        return False

    # def on_mouse_release(self, x, y, button):
    #     # send warps
    #     cell = self.dish.get_cell([x, y])
    #     cell.warps += 1000

    def step(self):
        # ----------------------------------------------------------------
        #
        # collect data
        #
        # ----------------------------------------------------------------

        if self.get_time() > 1:
            # pointer and target position frame
            pointer_x, pointer_y = self.follower.get_position()
            target_x, target_y = self.leader.get_position()

            distance = numpy.linalg.norm(self.leader.get_position() - self.follower.get_position())

            pointer_position = [self.get_time() - 1, pointer_x, pointer_y, target_x, target_y, distance,
                                self.follower.touch]
            pointer_frame = pandas.DataFrame([pointer_position], columns=self.pointer_columns)

            self.pointer_frame = pandas.concat([self.pointer_frame, pointer_frame])

        # actions frame (taken at each time step per cell)
        actions = [cell.action_taken for cell in self.followers] + \
                  [cell.action_taken for cell in self.leaders]

        action_frame = pandas.DataFrame([[self.get_time()] + actions], columns=["time"] + self.cells_columns)
        self.action_frame = pandas.concat([self.action_frame, action_frame])

        # ----------------------------------------------------------------
        #
        # plot action correlation
        #
        # ----------------------------------------------------------------

        green = ["cell-" + str(i) for i in range(0, len(self.followers))]
        red = ["runner-" + str(i) for i in range(0, len(self.leaders))]

        green_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[green], self.action_frame[green]).values

        red_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[red], self.action_frame[red]).values

        cross_correlation_mesh.data = \
            self.get_action_correlation(self.action_frame[red], self.action_frame[green]).values

        if self.screen.get_time() % 100 == 0:
            # print some stats
            print("[.] ---- time :", self.screen.get_time())

        if self.screen.get_time() % 1000 == 0:
            print("[.] ---- tally at time :", self.screen.get_time())
            self.pointer_frame.to_csv(self.output_folder + "pointer_frame.csv", index=False)
            self.action_frame.to_csv(self.output_folder + "action_frame.csv", index=False)
