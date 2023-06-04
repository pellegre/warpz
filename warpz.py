import argparse
import time
import numpy

import os
import imp
from warpz.simulation.local import *

from glumpy import collections, app, gl, gloo, data
from glumpy.geometry import primitives
from glumpy.transforms import Position, OrthographicProjection, Viewport, PanZoom

vertex_plotter = """
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

fragment_plotter = """
varying float v_intensity, v_color_r, v_color_g, v_color_b;

void main()
{
    vec3 color = vec3(v_color_r, v_color_g, v_color_b);
    gl_FragColor = vec4(color, v_intensity);
}
"""

vertex = """
attribute vec3 position;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    gl_Position = <transform(vec4(position.xy,0,1.0))>;
    v_texcoord = texcoord;
}
"""

fragment = """
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
    vec4 bg_color = vec4(colormap_gray(value), 1.0);
    gl_FragColor = bg_color;
} """

fragment_with_iso = """
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
    vec4 bg_color = vec4(colormap_icefire(value),1.0);
    vec4 fg_color = vec4(0,0,0,1);

    // Trace contour
    float levels = 16.0;

    float antialias = 1.0;
    float linewidth = 1.0 + antialias;
    if(length(value-0.5) < 0.5/levels)
        linewidth = 3.0 + antialias;

    float v  = levels*value - 0.5;
    float dv = linewidth/2.0 * fwidth(v);
    float f = abs(fract(v) - 0.5);
    float d = smoothstep(-dv,+dv,f);

    float t = linewidth/2.0 - antialias;
    d = abs(d)*linewidth/2.0 - t;
    if( d < 0.0 ) {
         gl_FragColor = bg_color;
    } else  {
        d /= antialias;
        gl_FragColor = mix(fg_color,bg_color,d);
    }
} """


class GlumpyVisualController:
    def __init__(self, windows_frame, plotter_frame, simulation):
        # windows and plotter
        self.window = windows_frame
        self.plotter = plotter_frame

        # simulation
        self.simulation = simulation
        self.last_agents_count = None
        self.last_vertex_count = None

        # update agent's positions (markers)
        self._update_agents_position()

        self.current_agents_count = len(self.agents_position)

        self.ortho_transform = OrthographicProjection(Position(), aspect=None)
        self.viewport = Viewport()

        self.markers = collections.MarkerCollection(marker='disc', transform=self.ortho_transform,
                                                    viewport=self.viewport)

        # attach to windows
        self.window.attach(self.ortho_transform)
        self.window.attach(self.viewport)

        self.segments = None
        if simulation.should_draw_vertex():
            self.current_vertex_count = len(self.source)
            self.segments = \
                collections.SegmentCollection('agg', transform=self.ortho_transform, viewport=self.viewport)

        # update markers
        self._update_markers()

        # meshing parameters
        self.mesh = self.simulation.get_mesh()
        if self.mesh is not None:
            # space program
            self.program = gloo.Program(vertex, fragment)

            scale = self.simulation.get_width() / self.simulation.get_height()
            self.vertices, self.indices = self._plane_primitive(x_size=2.0 * scale, y_size=2.0, n=64)
            self.program.bind(self.vertices)

            pan_zoom_transform = PanZoom(aspect=1)
            self.program['transform'] = pan_zoom_transform
            self.window.attach(pan_zoom_transform)

            # update space
            self._update_background_mesh()

        self.functions = {}

        if simulation.should_plot():
            for each in simulation.get_plots():
                self.functions[each] = gloo.Program(vertex_plotter, fragment_plotter, count=simulation.plotter_period)
                self.functions[each]["x"] = numpy.linspace(-1, 1, len(self.functions[each]))
                self.functions[each]["color_r"] = numpy.random.randint(100, 200) / 256
                self.functions[each]["color_g"] = numpy.random.randint(100, 200) / 256
                self.functions[each]["color_b"] = numpy.random.randint(100, 200) / 256

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

    def _update_agents_position(self):
        stride = [s if s > 0 else 1 for s in self.simulation.get_upper_bounds() - self.simulation.get_bottom_bounds()]
        scale = [self.simulation.get_width(), self.simulation.get_height(), 0]

        self.agents_position = (scale * (self.simulation.get_agents_position() -
                                         self.simulation.get_bottom_bounds())) / stride

        if self.simulation.should_draw_vertex():
            # set new vertex position on the screen
            self.source, self.target = numpy.nonzero(self.simulation.get_agents_vertex())
            self.current_vertex_count = len(self.source)

        self.current_agents_count = len(self.agents_position)

        self.agents_chromo = numpy.copy(self.simulation.get_chromo())
        self.agents_radius = numpy.copy(self.simulation.get_radius())

    def _update_mesh_values(self):
        if self.mesh is not None:
            self.mesh = self.simulation.get_mesh()

    def _update_background_mesh(self):
        if self.mesh is not None:
            self.program['data'] = (self.mesh - self.mesh.min()) / (self.mesh.max() - self.mesh.min())
            self.program['data'].interpolation = gl.GL_NEAREST
            self.program['data_shape'] = self.mesh.shape[1], self.mesh.shape[0]
            self.program['u_kernel'] = data.get("spatial-filters.npy")
            self.program['u_kernel'].interpolation = gl.GL_LINEAR

    def _update_markers(self):
        # initialize markers (one per agent)
        if self.last_agents_count is None and len(self.agents_position):
            self.markers.append(self.agents_position, size=self.agents_radius, linewidth=2, itemsize=1,
                                fg_color=(0, 0, 0, 1), bg_color=self.agents_chromo)

            if self.simulation.should_draw_vertex() and len(self.source) > 0:
                self.segments.append(self.agents_position[self.source], self.agents_position[self.target],
                                     linewidth=2, itemsize=1, color=(0, 0, 0, 1))

        self.last_agents_count = len(self.markers["position"])

        if self.last_agents_count > self.current_agents_count:
            sliced_agents = self.last_agents_count - self.current_agents_count

            self.agents_position = numpy.resize(self.agents_position, (self.last_agents_count, 3))

            self.agents_chromo = numpy.resize(self.agents_chromo, (self.last_agents_count, 4))
            self.agents_radius = numpy.resize(self.agents_radius, self.last_agents_count)

            self.agents_chromo[-sliced_agents:] = numpy.zeros((sliced_agents, 4))
            self.agents_radius[-sliced_agents:] = numpy.zeros(sliced_agents, dtype=int)

            self.last_agents_count = self.current_agents_count

        elif self.last_agents_count < self.current_agents_count:
            additional_agents = self.current_agents_count - self.last_agents_count

            self.agents_position = numpy.resize(self.agents_position, (self.last_agents_count, 3))

            self.agents_chromo = numpy.resize(self.agents_chromo, (self.last_agents_count, 4))
            self.agents_radius = numpy.resize(self.agents_radius, self.last_agents_count)

            agents_position = self.simulation.get_agents_position()

            self.markers.append(agents_position[-additional_agents:], size=self.agents_radius[-additional_agents:],
                                linewidth=1, itemsize=1, fg_color=(0, 0, 0, 1),
                                bg_color=self.agents_chromo[-additional_agents:])

            self.last_agents_count = self.current_agents_count

            self._update_agents_position()

        self.markers["position"] = self.agents_position

        if self.simulation.should_draw_vertex() and len(self.source) > 0:
            self.last_vertex_count = int(self.segments["P0"].shape[0] / 4)
            self.current_vertex_count = len(self.source)

            if self.last_vertex_count > self.current_vertex_count:

                self.source = numpy.resize(self.source, self.last_vertex_count)
                self.target = numpy.resize(self.target, self.last_vertex_count)

                self.last_vertex_count = self.current_vertex_count

            elif self.last_vertex_count < self.current_vertex_count:
                additional_vertices = self.current_vertex_count - self.last_vertex_count

                self.source = numpy.resize(self.source, self.current_vertex_count)
                self.target = numpy.resize(self.target, self.current_vertex_count)

                source, target = numpy.nonzero(self.simulation.get_agents_vertex())
                agents_position = self.simulation.get_agents_position()

                self.segments.append(agents_position[source[-additional_vertices:]],
                                     agents_position[target[-additional_vertices:]],
                                     linewidth=2, itemsize=1, color=(0, 0, 0, 1))

                self.last_vertex_count = self.current_vertex_count

            source_agents = numpy.repeat(self.agents_position[self.source], 4, axis=0)
            self.segments["P0"] = source_agents

            target_agents = numpy.repeat(self.agents_position[self.target], 4, axis=0)
            self.segments["P1"] = target_agents

            # self.segments["color"] = (self.agents_chromo[self.source] + self.agents_chromo[self.target]) / 2

        self.markers["bg_color"] = self.agents_chromo
        self.markers["size"] = self.agents_radius

    def on_timer(self, dt):
        # --- run simulation

        # run simulation step
        self.simulation.run()

        # display simulation information on the console
        self.simulation.step()

        # --- fetch values from the simulation

        # update agent's position
        self._update_agents_position()

        # update space values
        self._update_mesh_values()

        # --- update visual screen

        # set background space on program
        self._update_background_mesh()

        # set new markers position on the screen
        self._update_markers()

        # done with the simulation
        if self.simulation.done():
            exit(0)

    def on_draw(self, dt):
        self.window.clear()

        if self.mesh is not None:
            self.program.draw(gl.GL_TRIANGLES, self.indices)

        self.markers.draw()

        if self.simulation.should_draw_vertex() and len(self.source) > 0:
            self.segments.draw()

    def on_draw_plotter(self, dt):
        if self.simulation.should_plot():
            self.plotter.clear()

            plots = self.simulation.get_plots()
            for each in self.functions:
                if self.simulation.get_time() > 5:
                    self.functions[each].draw(gl.GL_LINE_STRIP)

                    index = (self.simulation.get_time() + 1) % len(self.functions[each])

                    self.functions[each]['intensity'] -= 1.0 / len(self.functions[each])
                    self.functions[each]['y'][index] = plots[each]
                    self.functions[each]['intensity'][index] = 1.0

    def resolve_coordinates(self, x, y):
        stride = [s if s > 0 else 1 for s in
                  self.simulation.get_upper_bounds() - self.simulation.get_bottom_bounds()]
        scale = [self.simulation.get_width(), self.simulation.get_height(), 0]
        height = self.simulation.get_height()

        position = self.simulation.get_bottom_bounds() + (numpy.array([x, height - y, 0]) * stride) / scale
        return numpy.array([p if not numpy.isnan(p) else 0 for p in position])

    def on_mouse_press(self, x, y, button):
        position = self.resolve_coordinates(x, y)
        if hasattr(self.simulation, "on_mouse_press"):
            getattr(self.simulation, "on_mouse_press")(position[0], position[1], button)

    def on_mouse_release(self, x, y, button):
        position = self.resolve_coordinates(x, y)
        if hasattr(self.simulation, "on_mouse_release"):
            getattr(self.simulation, "on_mouse_release")(position[0], position[1], button)

    def on_key_release(self, symbol, modifiers):
        if hasattr(self.simulation, "on_key_release"):
            getattr(self.simulation, "on_key_release")(symbol, modifiers)


def parse_args():
    # Parse input file with strategy definitions
    parser = argparse.ArgumentParser(description="visual warpz")
    parser.add_argument("-y", "--py", dest="input_file", action="store", required=True,
                        help="python input file where the environment is defined")
    parser.add_argument("-c", "--class", dest="class_name", action="store", required=False,
                        help="name of the environment class to load")
    # Return parsed arguments
    return parser.parse_args()


def instantiate_environment(file_path, expected_class):
    mod_name, file_ext = os.path.splitext(os.path.split(file_path)[-1])

    # Check on source file
    if file_ext.lower() == '.py':
        py_mod = imp.load_source(mod_name, file_path)

    # Check on compiled file
    elif file_ext.lower() == '.pyc':
        py_mod = imp.load_compiled(mod_name, file_path)

    environment = None
    if hasattr(py_mod, expected_class):
        environment = getattr(py_mod, expected_class)()

    if isinstance(environment, Simulation):
        print("[+] environment", expected_class, "successfully loaded from", file_path)
        return environment
    else:
        print("[+] couldn't load", expected_class, "from file", file_path)
        exit(1)


print("[+] visual warpz")
options = parse_args()
print("[+] reading class", options.class_name, "from file", options.input_file)

z = instantiate_environment(options.input_file, options.class_name)

window = app.Window(width=z.get_width(), height=z.get_height(), color=(1, 1, 1, 1))

if z.should_plot():
    plotter = app.Window(width=z.get_plotter_width(), height=z.get_plotter_height())


    @plotter.event
    def on_draw(dt):
        plotter.set_position(0, 0)
        glumpy_controller.on_draw_plotter(dt)

else:
    plotter = None

glumpy_controller = GlumpyVisualController(window, plotter, z)


@window.event
def on_draw(dt):
    window.set_position(0, 0)
    glumpy_controller.on_draw(dt)


@window.event
def on_mouse_press(x, y, button):
    glumpy_controller.on_mouse_press(x, y, button)


@window.event
def on_mouse_release(x, y, button):
    glumpy_controller.on_mouse_release(x, y, button)


@window.event
def on_key_release(symbol, modifiers):
    glumpy_controller.on_key_release(symbol, modifiers)


@window.timer(1.0 / 160.0)
def on_timer(dt):
    glumpy_controller.on_timer(dt)


app.run()
