from warpz.simulation.local import *
from warpz.space.box import *

from glumpy import app, collections
from glumpy import gl, gloo, data
from glumpy.geometry import primitives
from glumpy.transforms import Viewport, PanZoom
from glumpy import app, collections
from glumpy.graphics.text import FontManager
from glumpy.graphics.collections import GlyphCollection
from glumpy.transforms import Position, OrthographicProjection


import numpy

vertex = """
// Externs
// ------------------------------------
// extern vec3  position;
// extern float id;
// extern vec4  color;
// ... user-defined through collection init dtypes
// -----------------------------------------------
uniform float rows, cols;
varyin float v_x;
varyin vec4 v_color;
void main()
{
    // This line is mandatory and is responsible for fetching uniforms
    // from the underlyin uniform texture
    fetch_uniforms();
    // color can end up being an attribute or a varyin
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
// Collection varyins are not propagated to the fragment shader
// -------------------------------------------------------------
varyin float v_x;
varyin vec4 v_color;
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
varyin vec2 v_texcoord;
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
varyin vec2 v_texcoord;

void main()
{
    // Extract data value
    float value = Bicubic(data, data_shape, v_texcoord).r;
    // Map value to rgb color
    vec4 bg_color = vec4(colormap_hot(value), 1.0);
    gl_FragColor = bg_color;
} """


GRID = 50
SEED = 20
TILES = 5

COLORS = numpy.random.uniform(0.5, 1.0, (TILES, 4))

random_state = numpy.random.RandomState()

TIME_SCALE = 10000
WARPA_SCALE = 100
WARPA_FACTOR = int(TIME_SCALE / WARPA_SCALE)

observables_window = app.Window(1800, 1280)
yin_window = app.Window(700, 700)
yang_window = app.Window(700, 700)


class ForceMatrix:
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


class Observables:
    def __init__(self, rows=TILES, cols=1, points=TIME_SCALE):
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
            self.visitor.on_draw_distribution(self.positions)


# observables
observables = Observables()

# force meshes
yin_matrix = ForceMatrix(windows=yin_window, shape=(GRID, GRID))
yang_matrix = ForceMatrix(windows=yang_window, shape=(GRID, GRID))


@observables_window.event
def on_draw(dt):
    observables_window.set_position(600, 0)
    observables_window.set_title("observables " + str(dt))

    observables_window.clear()
    observables.lines.draw()

    observables.visit_on_draw()

    observables.positions[:, 0] = observables.positions[:, 1]
    observables.positions[:, -1] = observables.positions[:, -2]


@yin_window.event
def on_draw(dt):
    yin_window.set_position(600, 2000)

    yin_window.clear()

    yin_window.set_title("yin")

    yin_matrix.update_background_mesh()

    if yin_matrix.data is not None:
        yin_matrix.program.draw(gl.GL_TRIANGLES, yin_matrix.indices)


@yang_window.event
def on_draw(dt):
    yang_window.clear()
    yang_window.set_position(1700, 2000)

    yang_window.set_title("yang")

    yang_matrix.update_background_mesh()

    if yang_matrix.data is not None:
        yang_matrix.program.draw(gl.GL_TRIANGLES, yang_matrix.indices)


# ===========
#
# Matter - warps
#
# ===========

class Flow(Cell):
    def __init__(self, **kwargs):
        # initial stack
        self.flow = 0
        super().__init__(**kwargs)


# ===========
#
# Dharma - mediator
#
# ===========

class Dharma(Box):
    def __init__(self, grid=10, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Flow, **kwargs)

        self.dharma = numpy.zeros(self._dimensions * (self._grid,), int)
        self.entropy = numpy.zeros(shape=TIME_SCALE, dtype=float)

        self.forces, self.state = [], []

    def get_dharma(self):
        for each in self.get_children(condition=lambda c: isinstance(c, Matter)):
            self.dharma[each.index[1], each.index[0]] = each.flow

        return self.dharma

    def put(self, force):
        self.state = self.state + [len(self.forces)]
        self.forces = self.forces + [force]

        super().put(force)

    def __call__(self, signals):
        random_state.shuffle(self.state)

        one, two = self.forces[self.state[0]], self.forces[self.state[1]]

        # translocation
        source = one.select_source()
        target = two.get_cell(one.select_target(source).get_position())

        if source.warps > 0:
            one.distribution[source.warps] -= 1
            one.distribution[source.warps - 1] += 1

            two.distribution[target.warps] -= 1
            two.distribution[target.warps + 1] += 1

            assert one.distribution[source.warps] >= 0
            assert one.distribution[source.warps - 1] >= 0

            assert two.distribution[target.warps] >= 0
            assert two.distribution[target.warps + 1] >= 0

            assert sum(one.distribution) + sum(two.distribution) == 2 * (self.get_grid() ** 2)

            source.warps -= 1
            target.warps += 1

            self.dharma[source.index] += 1 if self.state[0] > 0 else -1

        signal_time = self.get_time() % TIME_SCALE
        if signal_time == 0:
            self.entropy[:] = 0

        self.entropy[signal_time] = ((one.entropy[signal_time] + two.entropy[signal_time]) -
                                     (one.entropy[0] + two.entropy[0]))


# ===========
#
# Matter - warps
#
# ===========

class Matter(Cell):
    def __init__(self, **kwargs):
        # initial stack
        self.warps = 0
        super().__init__(**kwargs)


# ===========
#
# Force
#
# ===========

class Force(Box):
    def __init__(self, seed=10, grid=10, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Matter, **kwargs)

        for each in self.get_children(condition=lambda c: isinstance(c, Matter)):
            each.warps = seed

        # matter underlyin this force
        self.matter = numpy.zeros(self._dimensions * (self._grid,), int)

        self.distribution = numpy.zeros(shape=14 * WARPA_SCALE, dtype=int)
        self.entropy = numpy.zeros(shape=TIME_SCALE, dtype=float)
        self.warps = numpy.zeros(shape=TIME_SCALE, dtype=float)

        self.distribution[seed] = grid * grid

    def get_matter(self):
        for each in self.get_cells():
            self.matter[each.index[1], each.index[0]] = each.warps

        return self.matter

    @staticmethod
    def log_factorial(n):
        if n > 0:
            return n * math.log(n) - n
        return 0

    @staticmethod
    def log_factorial__(n):
        if n > 0:
            return sum([math.log(x) for x in range(1, n + 1)])
        return 0

    def select_source(self):
        x, y = random_state.uniform(0.0, 1.0), random_state.uniform(0.0, 1.0)
        return self.get_cell([x, y])

    def select_target(self, source):
        raise EnvironmentError(self)

    def __call__(self, signals):
        source = self.select_source()
        target = self.select_target(source)

        if source.warps > 0 and source is not target:
            self.distribution[source.warps] -= 1
            self.distribution[source.warps - 1] += 1

            self.distribution[target.warps] -= 1
            self.distribution[target.warps + 1] += 1

            assert self.distribution[source.warps] >= 0
            assert self.distribution[source.warps - 1] >= 0

            assert self.distribution[target.warps] >= 0
            assert self.distribution[target.warps + 1] >= 0

            source.warps -= 1
            target.warps += 1

        z = self.log_factorial(sum(self.distribution))
        x = sum([self.log_factorial(x) for x in self.distribution])

        signal_time = self.get_time() % TIME_SCALE
        if signal_time == 0:
            self.entropy[:] = 0
            self.warps[:] = 0

        self.entropy[signal_time] = z - x
        self.warps[signal_time] = sum(self.get_matter().flatten())


class Yin(Force):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_target(self, source):
        neighbors = self.get_periodic_neighbors(source) + [source]
        target = neighbors[random_state.randint(0, len(neighbors))]

        return target


class Yang(Force):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def select_target(self, source):
        x, y = random_state.uniform(0.0, 1.0), random_state.uniform(0.0, 1.0)
        return self.get_cell([x, y])


# ===========
#
# Tao Simulation
#
# ===========


class Tao(Simulation):
    def __init__(self):
        self.dharma = Dharma(grid=GRID)

        self.yin = Yin(grid=GRID, seed=SEED)
        self.yang = Yang(grid=GRID, seed=SEED)

        z = [self.yin, self.yang]
        # random_state.shuffle(z)

        self.dharma.put(z[0])
        self.dharma.put(z[1])

        observables.hook_visitor(self)
        yin_matrix.hook_visitor(self)
        yang_matrix.hook_visitor(self)

        super().__init__(universe=self.dharma, agents_filter=lambda agent: False,
                         min_radius=9, max_radius=14)

    def on_draw_distribution(self, positions):
        max_value = numpy.max(numpy.fabs(self.dharma.entropy))
        if max_value > 0:
            positions[0, 1:-1, 1] = 0.8 * self.dharma.entropy / max_value
        else:
            positions[0, 1:-1, 1] = 0

        one_distribution = self.dharma.forces[0].distribution[:] / numpy.max(self.dharma.forces[0].distribution[:])
        for i in range(0, WARPA_SCALE):
            positions[1, i * WARPA_FACTOR:(i + 1) * WARPA_FACTOR, 1] = one_distribution[i]

        two_distribution = self.dharma.forces[1].distribution[:] / numpy.max(self.dharma.forces[1].distribution[:])
        for i in range(0, WARPA_SCALE):
            positions[2, i * WARPA_FACTOR:(i + 1) * WARPA_FACTOR, 1] = two_distribution[i]

        entropy_flow = self.dharma.forces[0].entropy - self.dharma.forces[1].entropy
        max_value = numpy.max(numpy.fabs(entropy_flow))
        if max_value > 0:
            positions[3, 1:-1, 1] = 0.8 * entropy_flow / max_value
        else:
            positions[3, 1:-1, 1] = 0

        warps_flow = self.dharma.forces[0].warps - self.dharma.forces[1].warps
        max_value = numpy.max(numpy.fabs(warps_flow))
        if max_value > 0:
            positions[4, 1:-1, 1] = 0.8 * warps_flow / max_value
        else:
            positions[4, 1:-1, 1] = 0

    def should_plot(self):
        return False

    def get_mesh(self):
        return self.dharma.get_dharma()

    def done(self):
        return False

    def step(self):
        if self.dharma.get_time() % 100 == 0:
            # print some stats
            print("[.][.][.][.][.][.][.][.][%][.][.][.][.][.][.][.][.][.]")

            print("[.] time          :", self.dharma.get_time())

            signal_time = self.get_time() % TIME_SCALE - 1
            entropy_flow = self.dharma.forces[0].entropy[signal_time] - self.dharma.forces[1].entropy[signal_time]

            print(f"[$] entropy flow  : {entropy_flow:.2f}")
            print(f"[$] entropy       : {self.dharma.entropy[signal_time]:.2f}")

            one_warps = sum(self.dharma.forces[0].get_matter().flatten())
            print("[@] warps (one)   :", one_warps, type(self.dharma.forces[0]).__name__)

            two_warps = sum(self.dharma.forces[1].get_matter().flatten())
            print("[@] warps (two)   :", two_warps, type(self.dharma.forces[1]).__name__)

            print("[@] warps (total) :", one_warps + two_warps)
            print("[@] dharma        :", sum(self.dharma.dharma.flatten()))

        for force in self.dharma.forces:
            if isinstance(force, Yin):
                yin_matrix.data = force.get_matter()
            elif isinstance(force, Yang):
                yang_matrix.data = force.get_matter()
