import numpy
import math
import copy

from bisect import bisect

from glumpy import collections, app, gl, gloo, data
from glumpy.geometry import primitives
from glumpy.transforms import Position, OrthographicProjection, Viewport, PanZoom
from glumpy import app
from glumpy.graphics.text import FontManager
from glumpy.transforms import Position, Trackball, Viewport
from glumpy.graphics.collections import GlyphCollection
from glumpy.graphics.collections import PathCollection
from glumpy.graphics.collections import SegmentCollection

vertex_function_plotter = """
attribute float x, y, intensity, color_r, color_g, color_b;
varying float v_intensity, v_color_r, v_color_g, v_color_b;
void main (void)
{
    v_intensity = intensity;
    v_color_r = color_r; 
    v_color_g = color_g;
    v_color_b = color_b;
    gl_Position = vec4(x, y, 0, 1.3);
}
"""

fragment_function_plotter = """
varying float v_intensity, v_color_r, v_color_g, v_color_b;

void main()
{
    vec3 color = vec3(v_color_r, v_color_g, v_color_b);
    gl_FragColor = vec4(color, v_intensity);
}
"""


class Plotter:
    def __init__(self, plotter_frame, functions, plotter_period, title=None, log_scale=False):
        self.functions = dict()
        self.plotter = plotter_frame
        self.plotter_period = plotter_period

        self.title = title
        self.log_scale = log_scale

        for each in functions:
            self.functions[each] = gloo.Program(vertex_function_plotter, fragment_function_plotter,
                                                count=self.plotter_period)
            self.functions[each]["x"] = numpy.linspace(-1, 1, len(self.functions[each]))
            self.functions[each]["color_r"] = numpy.random.uniform(0.50, 1.00)
            self.functions[each]["color_g"] = numpy.random.uniform(0.50, 1.00)
            self.functions[each]["color_b"] = numpy.random.uniform(0.50, 1.00)

        self.time, self.plots, self.plot_scale = 0, dict(), 1

    @staticmethod
    def scale_plot(value, scale_factor):
        return 1.7 * (value / scale_factor - 0.5)

    @staticmethod
    def unscale_plot(value, scale_factor):
        return ((value / 1.7) + 0.5) * scale_factor

    def update_data(self, time, points):
        self.time = time

        for each in points:
            if each in self.functions:
                self.plots[each] = points[each]
            else:
                raise RuntimeError("invalid point " + str(each))

    def on_draw(self, dt):
        self.plotter.clear()

        gl.glClearColor(0, 0, 0, 1)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if len(self.plots):
            max_value = max(self.plots.values())
            if self.log_scale:
                max_value = math.log(max_value) if max_value > 0 else 0

            values = str()
            for each in self.plots:
                values += each + "=" + f"{self.plots[each]:.2f} "

            if self.title is not None:
                self.plotter.set_title(values)

            if max_value > self.plot_scale:
                for each in self.functions:
                    for i in range(0, len(self.functions[each])):
                        value = self.functions[each]['y'][i]
                        self.functions[each]['y'][i] = \
                            self.scale_plot(self.unscale_plot(value, self.plot_scale), max_value)

                self.plot_scale = max_value

            for each in self.functions:
                if self.time > 5:
                    self.functions[each].draw(gl.GL_LINE_STRIP)

                    index = (self.time + 1) % len(self.functions[each])

                    self.functions[each]['intensity'] -= 1.0 / len(self.functions[each])
                    if self.log_scale:
                        value = math.log(self.plots[each]) if self.plots[each] > 0 else 0
                    else:
                        value = self.plots[each]

                    self.functions[each]['y'][index] = self.scale_plot(value, self.plot_scale)

                    self.functions[each]['intensity'][index] = 1.0

            self.time += 1


vertex_histogram = """
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

fragment_histogram = """
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


class HistogramPlotter:
    def __init__(self, plotter_frame, rows, cols, points, segment):
        # plotter frame
        self.plotter = plotter_frame

        # plot matrix
        self.rows, self.cols = rows, cols
        self.factor, self.bins = 10, points
        self.n, self.points = rows * cols, self.factor * points

        # histogram segments and bins
        self.segment = segment

        # signal lines
        self.lines = collections.RawPathCollection(
            user_dtype=[("amplitude", (numpy.float32, 1), 'shared', 1),
                        ("selected", (numpy.float32, 1), 'shared', 0),
                        ("xscale", (numpy.float32, 1), 'shared', 1)],
            color="shared", vertex=vertex_histogram, fragment=fragment_histogram)

        self.lines.append(numpy.zeros((self.n * self.points, 3)), itemsize=self.points)

        self.lines["rows"] = self.rows
        self.lines["cols"] = self.cols
        self.lines["amplitude"][:self.n] = 1.0
        self.lines["color"][:self.n] = numpy.random.uniform(0.5, 1.0, (rows * cols, 4))
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

        # create histogram matrix
        self.space = numpy.linspace(self.segment[0], self.segment[1], self.bins, endpoint=False)
        self.histogram, self.count = numpy.zeros((rows, cols, self.bins), dtype=int), 0

    def add(self, matrix):
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                index = bisect(self.space, matrix[i][j]) - 1
                self.histogram[i, j, index] += 1

        self.count += 1

    def on_draw(self, dt):
        count = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                # get max value
                max_value = self.histogram[i][j].max()

                # normalize bin values
                for k in range(0, self.bins):
                    idx = k * self.factor

                    # get histogram value
                    value = self.histogram[i][j][k] / max_value
                    self.positions[count, idx:idx + self.factor, 1] = 2 * value - 1

                # count windows
                count += 1

        self.lines.draw()

        # Here we ensure:
        #   * first point = second point
        #   * last point = prev last point
        self.positions[:, 0] = self.positions[:, 1]
        self.positions[:, -1] = self.positions[:, -2]


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


class PlotMesh:
    @staticmethod
    def _plane_primitive(x_size=1.0, y_size=1.0, n=2):
        """
        Plane centered at origin, lying on the XY-plane

        Parameters
        ----------
        x_size : float
           plane length size

        y_size : float
           plane length size

        n : int
            Tesselation level
        """

        n = max(2, n)

        T = numpy.linspace(0, 1, n, endpoint=True)
        X, Y = numpy.meshgrid(T - 0.5, T - 0.5)
        X = X.ravel() * x_size
        Y = Y.ravel() * y_size
        U, V = numpy.meshgrid(T, T)
        U = U.ravel()
        V = V.ravel()

        I = (numpy.arange((n - 1) * n, dtype=numpy.uint32).reshape(n - 1, n))[:, :-1].T
        I = numpy.repeat(I.ravel(), 6).reshape(n - 1, n - 1, 6)
        I[:, :] += numpy.array([0, 1, n + 1, 0, n + 1, n], dtype=numpy.uint32)

        vtype = [('position', numpy.float32, 3),
                 ('texcoord', numpy.float32, 2),
                 ('normal', numpy.float32, 3)]
        itype = numpy.uint32

        vertices = numpy.zeros((6, n * n), dtype=vtype)
        vertices["texcoord"][..., 0] = U
        vertices["texcoord"][..., 1] = V
        vertices["position"][0, :, 0] = X
        vertices["position"][0, :, 1] = Y
        vertices["position"][0, :, 2] = 0
        vertices["normal"][0] = 0, 0, 1

        vertices = vertices.ravel()
        indices = numpy.array(I, dtype=itype).ravel()
        return vertices.view(gloo.VertexBuffer), indices.view(gloo.IndexBuffer)

    def __init__(self, windows, shape):
        self.windows = windows

        # space program
        self.program = gloo.Program(mesh_vertex, mesh_fragment)
        scale = shape[1] / shape[0]
        self.vertices, self.indices = self._plane_primitive(x_size=2.0 * scale, y_size=2.0, n=64)

        self.program.bind(self.vertices)

        self.pan_zoom_transform = PanZoom(aspect=1)
        self.program['transform'] = self.pan_zoom_transform
        self.windows.attach(self.pan_zoom_transform)

        # mesh
        self.data = numpy.zeros(shape=shape)
        self.min, self.max = 0.00, 1.00

    def update_background_mesh(self):
        if self.data is not None:
            delta = 1 if self.max == self.min else self.max - self.min
            self.program['data'] = (self.data - self.min) / delta
            self.program['data'].interpolation = gl.GL_NEAREST
            self.program['data_shape'] = self.data.shape[1], self.data.shape[0]
            self.program['u_kernel'] = data.get("spatial-filters.npy")
            self.program['u_kernel'].interpolation = gl.GL_LINEAR

    def set_mesh(self, matrix, minimum=None, maximum=None):
        # copy data
        self.data = matrix

        # set minimum
        if minimum is None:
            self.min = self.data.min()

        # set maximum
        if maximum is None:
            self.max = self.data.max()

    def on_draw(self, dt):
        # update mesh
        self.update_background_mesh()

        if self.data is not None:
            self.program.draw(gl.GL_TRIANGLES, self.indices)


vertex_tiles = """
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

fragment_tiles = """
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


class TilesPlotter:
    def __init__(self, plotter_frame, rows, cols, points=4098, time_factor=1):
        # plotter frame
        self.plotter = plotter_frame

        # plot matrix
        self.rows, self.cols = rows, cols
        self.time_factor, self.time_scale = time_factor, points
        self.n, self.points = rows * cols, points

        # signal lines
        self.lines = collections.RawPathCollection(
            user_dtype=[("amplitude", (numpy.float32, 1), 'shared', 1),
                        ("selected", (numpy.float32, 1), 'shared', 0),
                        ("xscale", (numpy.float32, 1), 'shared', 1)],
            color="shared", vertex=vertex_histogram, fragment=fragment_histogram)

        self.lines.append(numpy.zeros((self.n * self.points, 3)), itemsize=self.points)

        self.lines["rows"] = self.rows
        self.lines["cols"] = self.cols
        self.lines["amplitude"][:self.n] = 1.0
        self.lines["color"][:self.n] = numpy.random.uniform(0.5, 1.0, (rows * cols, 4))
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

        # time step
        self.time = 0

    def add(self, time, matrix):
        # update time
        self.time = time

        # re-scale signal time
        signal_time = (self.time * self.time_factor) % self.time_scale

        # reset plot
        if signal_time == 0:
            self.positions[:, 1:-1, 1] = 0

        # update data
        count = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.positions[count, signal_time:signal_time + 1, 1] = matrix[i][j]

                # update counter
                count += 1

    def push(self, matrix):
        # re-scale signal time
        signal_time = (self.time * self.time_factor) % self.time_scale

        # reset plot
        if signal_time == 0:
            self.positions[:, 1:-1, 1] = 0

        # update data
        count = 0
        for i in range(0, self.rows):
            for j in range(0, self.cols):
                self.positions[count, 1:-1, 1] = matrix[i][j][:]

                # update counter
                count += 1

    def on_draw(self, dt):
        # draw lines
        self.lines.draw()

        # update
        self.positions[:, 0] = self.positions[:, 1]
        self.positions[:, -1] = self.positions[:, -2]


class Graph:
    def __init__(self, x_limits, y_limits, title=str(), x_title=str(), y_title=str(),
                 width=2500, height=1200, position=(1330, 0), linewidth=10):
        # windows position
        self.position = position

        # line width
        self.linewidth = linewidth

        # windows plot
        self.window = app.Window(width=width, height=height, color=(1, 1, 1, 1))

        # axis bins
        self.bins = 10

        # regular
        self.regular = FontManager.get("OpenSans-Regular.ttf")
        self.scale = 0.001

        # store title
        self.x_title, self.y_title, self.title = x_title, y_title, title

        # absolute limits
        self.x_min, self.x_max = -2.10, 2.30
        self.y_min, self.y_max = -0.90, 1.00

        # relative limits
        self.x_val_min, self.x_val_max = x_limits[0], x_limits[1]
        self.y_val_min, self.y_val_max = y_limits[0], y_limits[1]

        # fix z
        self.z = 0

        # graph
        self.labels = None
        self.transform = Trackball(Position())
        self.viewport = Viewport()
        self.ticks = SegmentCollection(mode="agg", transform=self.transform, viewport=self.viewport,
                                       linewidth="local", color="local")

        # set scale
        self.set_scale()

        # set ticks
        self.set_ticks()

        # reset transform
        self.reset_transform()

        # tickers
        self.tickers = dict()

        # current label height
        self.label_height = 0.90

    def add_ticker(self, name, color):
        # add ticker
        self.tickers[name] = dict()
        self.tickers[name]["color"] = color

        # set label
        self.tickers[name]["legend"] = GlyphCollection(transform=self.transform, viewport=self.viewport)
        self.tickers[name]["legend"].append(name, self.regular, origin=(-1.80, self.label_height, self.z),
                                            scale=1.5 * self.scale, direction=(1, 0, 0),
                                            anchor_x="left", anchor_y="center")

        # add ticker label path
        path = PathCollection(mode="agg+", transform=self.transform, viewport=self.viewport)
        for i, y in enumerate(numpy.linspace(self.label_height - 0.02, self.label_height + 0.01, 100)):
            path.append(numpy.array([[-1.95, y, 0], [-1.85, y, 0]]), color=color, closed=False)
            path["color"] = color

        # store label path
        self.tickers[name]["label"] = path
        self.window.attach(self.tickers[name]["label"]["transform"])
        self.window.attach(self.tickers[name]["label"]["viewport"])

        # add ticker path
        self.tickers[name]["graph"] = PathCollection(mode="agg+", transform=self.transform, viewport=self.viewport)
        self.tickers[name]["graph"]["color"] = color

        # attach it
        self.window.attach(self.tickers[name]["graph"]["transform"])
        self.window.attach(self.tickers[name]["graph"]["viewport"])

        # data points (initialized in origin)
        self.tickers[name]["orig"] = list()
        self.tickers[name]["data"] = list()

        # update label height
        self.label_height -= 0.10

        # reset transform
        self.reset_transform()

    def add_point(self, name, x, y):
        # shift x limits when plot is out of scale
        if x > self.x_val_max:
            self.reset_x_limits()

        # increase y upper limit when graph is out of scale
        if y > self.y_val_max:
            self.reset_y_limits(value=y)

        # calculate scale
        fx = ((x - self.x_val_min) / (self.x_val_max - self.x_val_min))
        fy = ((y - self.y_val_min) / (self.y_val_max - self.y_val_min))

        # calculate x and y values
        x_val, y_val = self.x_min + fx * (self.x_max - self.x_min), self.y_min + fy * (self.y_max - self.y_min)

        # append data point
        self.tickers[name]["orig"].append([x, y, 0])
        self.tickers[name]["data"].append([x_val, y_val, 0])

        # draw it
        for i in range(0, self.linewidth):
            if len(self.tickers[name]["data"]) >= 2:
                self.tickers[name]["graph"].append(numpy.array(self.tickers[name]["data"][-2:]),
                                                   color=self.tickers[name]["color"], closed=False)

    def set_ticks(self):
        # frame
        # -------------------------------------
        pl = [(self.x_min, self.y_min, self.z), (self.x_min, self.y_max, self.z),
              (self.x_max, self.y_max, self.z), (self.x_max, self.y_min, self.z)]
        pr = [(self.x_min, self.y_max, self.z), (self.x_max, self.y_max, self.z),
              (self.x_max, self.y_min, self.z), (self.x_min, self.y_min, self.z)]
        self.ticks.append(pl, pr, linewidth=2)

        # grids
        # -------------------------------------
        n = self.bins + 1
        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))

        pl[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pl[:, 1] = self.y_min
        pl[:, 2] = self.z
        pr[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pr[:, 1] = self.y_max
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1, color=(0, 0, 0, 0.25))

        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))
        pl[:, 0] = self.x_min
        pl[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pl[:, 2] = self.z
        pr[:, 0] = self.x_max
        pr[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1, color=(0, 0, 0, 0.25))

        # majors
        # -------------------------------------
        n = self.bins + 1
        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))
        pl[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pl[:, 1] = self.y_min - 0.015
        pl[:, 2] = self.z
        pr[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pr[:, 1] = self.y_min + 0.025 * (self.y_max - self.y_min)
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1.5)
        pl[:, 1] = self.y_max + 0.015
        pr[:, 1] = self.y_max - 0.025 * (self.y_max - self.y_min)
        self.ticks.append(pl, pr, linewidth=1.5)

        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))
        pl[:, 0] = self.x_min - 0.015
        pl[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pl[:, 2] = self.z
        pr[:, 0] = self.x_min + 0.025 * (self.x_max - self.x_min)
        pr[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1.5)
        pl[:, 0] = self.x_max + 0.015
        pr[:, 0] = self.x_max - 0.025 * (self.x_max - self.x_min)
        self.ticks.append(pl, pr, linewidth=1.5)

        # minors
        # -------------------------------------
        n = self.bins * self.bins + self.bins + 1
        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))
        pl[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pl[:, 1] = self.y_min
        pl[:, 2] = self.z
        pr[:, 0] = numpy.linspace(self.x_min, self.x_max, n)[1:-1]
        pr[:, 1] = self.y_min + 0.0125 * (self.y_max - self.y_min)
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1)
        pl[:, 1] = self.y_max
        pr[:, 1] = self.y_max - 0.0125 * (self.y_max - self.y_min)
        self.ticks.append(pl, pr, linewidth=1)

        pl = numpy.zeros((n - 2, 3))
        pr = numpy.zeros((n - 2, 3))
        pl[:, 0] = self.x_min
        pl[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pl[:, 2] = self.z
        pr[:, 0] = self.x_min + 0.0125 * (self.x_max - self.x_min)
        pr[:, 1] = numpy.linspace(self.y_min, self.y_max, n)[1:-1]
        pr[:, 2] = self.z
        self.ticks.append(pl, pr, linewidth=1)
        pl[:, 0] = self.x_max
        pr[:, 0] = self.x_max - 0.0125 * (self.x_max - self.x_min)
        self.ticks.append(pl, pr, linewidth=1)

    def set_scale(self):
        self.labels = GlyphCollection(transform=self.transform, viewport=self.viewport)

        # number of bins (majors) in the axis
        n = self.bins + 1

        # y axis values
        for i, y in enumerate(numpy.linspace(self.y_min, self.y_max, n)):
            text = "%d" % int(self.y_val_min + i * ((self.y_val_max - self.y_val_min) / self.bins))
            self.labels.append(text, self.regular, origin=(-2.20, y, self.z), scale=self.scale, direction=(1, 0, 0),
                               anchor_x="center", anchor_y="top")

        # x axis values
        for i, x in enumerate(numpy.linspace(self.x_min, self.x_max, n)):
            text = "%d" % int(self.x_val_min + i * ((self.x_val_max - self.x_val_min) / self.bins))
            self.labels.append(text, self.regular, origin=(x, -0.97, self.z), scale=self.scale, direction=(1, 0, 0),
                               anchor_x="center", anchor_y="top")

        # set x title
        self.labels.append(self.x_title, self.regular, origin=(0.00, -1.10, self.z), scale=1.5 * self.scale,
                           direction=(1, 0, 0),
                           anchor_x="center", anchor_y="center")

        # set x title
        self.labels.append(self.x_title, self.regular, origin=(0.00, -1.10, self.z), scale=1.5 * self.scale,
                           direction=(1, 0, 0),
                           anchor_x="center", anchor_y="center")

        # set y title
        self.labels.append(self.y_title, self.regular, origin=(-2.35, 0.00, self.z), scale=1.5 * self.scale,
                           direction=(0, 1, 0),
                           anchor_x="center", anchor_y="center")

        # set main title
        self.labels.append(self.title, self.regular, origin=(0.00, 1.10, self.z), scale=1.7 * self.scale,
                           direction=(1, 0, 0),
                           anchor_x="center", anchor_y="center")

    def reset_transform(self):
        self.transform.theta = 0
        self.transform.phi = 0
        self.transform.zoom = 16.5

    def reset_tickers(self):
        # set scale
        self.set_scale()

        # reset tickers
        for each in self.tickers:
            # add ticker path
            self.tickers[each]["graph"] = PathCollection(mode="agg+", transform=self.transform, viewport=self.viewport)
            self.tickers[each]["graph"]["color"] = self.tickers[each]["color"]

            # attach it
            self.window.attach(self.tickers[each]["graph"]["transform"])
            self.window.attach(self.tickers[each]["graph"]["viewport"])

            # data points (initialized in origin)
            self.tickers[each]["orig"] = list()
            self.tickers[each]["data"] = list()

        # reset transform
        self.reset_transform()

    def reset_x_limits(self):
        # shift x axis
        x_val_min, x_val_max = self.x_val_min, self.x_val_max
        self.x_val_min, self.x_val_max = x_val_max, x_val_max + (x_val_max - x_val_min)

        # reset tickers
        self.reset_tickers()

    def reset_y_limits(self, value):
        # shift x axis
        self.y_val_max = int(value) + (10 - int(value) % 10)

        # store already plotted values
        orig = {name: copy.deepcopy(self.tickers[name]["orig"]) for name in self.tickers}

        # reset tickers
        self.reset_tickers()

        # add and re scale points
        for name in orig:
            for point in orig[name]:
                self.add_point(name, point[0], point[1])

    def on_draw(self, dt):
        self.window.set_position(self.position[0], self.position[1])

        # clear windows
        self.window.clear()

        # draw ticks and labels
        self.ticks.draw()
        self.labels.draw()

        # draw tickers
        for name in self.tickers:
            self.tickers[name]["legend"].draw()
            self.tickers[name]["label"].draw()

            # draw graph
            if len(self.tickers[name]["data"]) >= 2:
                self.tickers[name]["graph"].draw()
