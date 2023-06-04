import string
import subprocess
import tempfile
import random
import os
import time

from shutil import which
from networkx.drawing.nx_agraph import to_agraph


class Plotter:
    def __init__(self):
        pass

    @staticmethod
    def _get_tmp_filename():
        return tempfile.gettempdir() + "/" + "".join(random.choice(string.ascii_letters) for _ in range(12))

    @staticmethod
    def _get_latex_node(state):
        if "_" in state:
            split = state.split("_")
            prefix, number = split[0], "%s" % "_ ".join(map(str, split[1:]))
        else:
            prefix, number = state, ""

        if prefix in ["nu", "lambda"]:
            prefix = '\\' + prefix

        if "_" in number:
            return str(prefix) + "_{" + Plotter._get_latex_node(number) + "}"

        return str(prefix) + "_{" + number + "}"

    @staticmethod
    def show(machine, tex=True, layout="dot"):
        a = to_agraph(machine.g)
        a.graph_attr["rankdir"] = "LR"
        a.graph_attr["size"] = 8.5
        a.node_attr["fontsize"] = 7
        a.edge_attr["fontsize"] = 8
        a.edge_attr["penwidth"] = 0.4
        a.edge_attr["arrowsize"] = 0.5

        if tex:
            a.node_attr["texmode"] = "math"
            a.edge_attr["texmode"] = "math"

        for state in machine.states:
            node = a.get_node(state.symbol)
            if tex:
                node.attr["label"] = Plotter._get_latex_node(state.symbol)

            if state in machine.final:
                node.attr["shape"] = "doublecircle"

        a.layout(layout)

        edges = {source.symbol: dict() for source in machine.output_flux}
        for source in machine.output_flux:
            for target in machine.output_flux[source]:
                for flow in machine.output_flux[source][target]:
                    ei, ef = source.symbol, target.symbol

                    if flow.flux is None:
                        if ef not in edges[ei]:
                            edges[ei][ef] = flow.symbol
                        else:
                            edges[ei][ef] = edges[ei][ef] + "," + flow.symbol

                    else:
                        a.add_node(flow.symbol, label=Plotter._get_latex_node(flow.symbol), style="invisible")
                        edges[flow.symbol] = dict()

                        a.remove_edge(ei, ef)

                        # middle symbol
                        edges[ei][flow.symbol] = flow.symbol
                        a.add_edge(ei, flow.symbol)

                        edges[flow.symbol][ef] = flow.symbol
                        a.add_edge(flow.symbol, ef)

        hidden_count = 0
        for state in machine.states.union(machine.flows):
            if state.flux is not None:
                for flow in state.flux.flows:
                    hidden_node = "hidden" + str(hidden_count)

                    # add hidden node
                    a.add_node(hidden_node, style="invisible")
                    hidden_count += 1

                    edges[state.symbol][hidden_node] = flow.symbol
                    a.add_edge(state.symbol, hidden_node, color="red")

        for ei in edges:
            for ef in edges[ei]:
                symbol = edges[ei][ef]
                edge = a.get_edge(ei, ef)

                if tex:
                    edge.attr["texlbl"] = "$" + Plotter._get_latex_node(symbol) + "$"
                    edge.attr["label"] = "  "
                else:
                    edge.attr["label"] = symbol

        if tex:
            temp_dir = tempfile.gettempdir()
            filename = Plotter._get_tmp_filename()

            dot_file = filename + ".dot"
            tex_file = filename + ".tex"

            if which("dot2tex") is not None:
                with open(dot_file, "w") as f:
                    f.write(a.to_string())
                    subprocess.Popen(["dot2tex --crop -ftikz " + dot_file + " > " +
                                      tex_file + " && pdflatex --output-directory " + temp_dir +
                                      " 2>1 >> /dev/null " + tex_file], shell=True)
                    f.close()

                    # final file
                    filename = filename + ".pdf"

                    # wait for file to be crafted
                    time_to_wait = 10
                    time_counter = 0
                    while not os.path.exists(filename):
                        time.sleep(1)
                        time_counter += 1
                        if time_counter > time_to_wait:
                            raise RuntimeError("file " + filename + " was not crafted")

            else:
                raise RuntimeError("dot2tex not installed")

        else:
            filename = Plotter._get_tmp_filename() + ".pdf"
            a.draw(path=filename)

        if which("xdg-open") is not None:
            subprocess.Popen(["xdg-open " + filename], shell=True)
