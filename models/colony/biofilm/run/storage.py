from models.colony.biofilm.species import *
from models.plotter.window import *

import pandas
import json
import pickle
import base64
import os
import sys
import shutil


class Coco:
    def __init__(self, coconata):
        # agent index
        self.idx = coconata.get_id()

        # substrate container
        self.container = coconata.get_container()

        # stress state
        self.unbalanced, self.starving, self.scarce, self.planktonic = False, False, False, False

        # time step
        self.time = coconata.get_time()

        # position
        self.position = coconata.get_position()

        # genes
        self.genes = coconata.get_tallies()


class CocoStep:
    def __init__(self, medium):
        # medium
        self.medium = medium
        self.columns = ["idx", "time", "container", "x", "y", "unbalanced", "starving", "scarce", "planktonic", "genes"]

        # table
        self.tables = list()

    def parse_step(self):
        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        # coco table
        table = {c: list() for c in self.columns}

        # get cocos
        for coco in coconatas:
            coconata = Coco(coconata=coco)

            table["idx"].append(coconata.idx)
            table["time"].append(coconata.time)
            table["container"].append(json.dumps(coconata.container))
            table["x"].append(coconata.position[0])
            table["y"].append(coconata.position[1])
            table["unbalanced"].append(coconata.unbalanced)
            table["starving"].append(coconata.starving)
            table["scarce"].append(coconata.scarce)
            table["planktonic"].append(coconata.planktonic)
            table["genes"].append(json.dumps(coconata.genes))

        self.tables.append(pandas.DataFrame(table, columns=self.columns))

    def get_table(self):
        return pandas.concat(self.tables)


class BlockStep:
    def __init__(self, medium):
        # medium
        self.medium = medium
        self.columns = ["idx", "idy", "time", "container", "flux"]

        # table
        self.tables = list()

    def parse_step(self):
        # cellulatas
        cells = list(self.medium.get_agents(condition=lambda a: isinstance(a, Block)))

        # coco table
        table = {c: list() for c in self.columns}

        # get cocos
        for cell in cells:
            table["idx"].append(cell.index[0])
            table["idy"].append(cell.index[1])
            table["time"].append(cell.get_time())

            container = cell.get_container()
            table["container"].append(json.dumps({s: float(container[s]) for s in container}))
            table["flux"].append(cell.get_gene_flux())

        self.tables.append(pandas.DataFrame(table, columns=self.columns))

    def get_table(self):
        return pandas.concat(self.tables)


class UniverseStep:
    def __init__(self, medium):
        # medium
        self.medium = medium

        # universe bounds
        self.bottom_bounds = medium.get_bottom_bounds()
        self.upper_bounds = medium.get_upper_bounds()


class GeneStep:
    def __init__(self, medium):
        # medium
        self.medium = medium
        self.columns = ["idx", "time", "chromosome"]

        # table
        self.tables = list()

    def parse_step(self):
        # cellulatas
        coconatas = list(self.medium.get_agents(condition=lambda a: isinstance(a, Particle)))

        # coco table
        table = {c: list() for c in self.columns}

        # get cocos
        for coco in coconatas:
            table["idx"].append(coco.get_id())
            table["time"].append(coco.get_time())

            chromosome = base64.b64encode(pickle.dumps(coco.get_chromosome()))
            table["chromosome"].append(chromosome)

        self.tables.append(pandas.DataFrame(table, columns=self.columns))

    def get_table(self):
        return pandas.concat(self.tables)


class BioDriver:
    def __init__(self, filename, biofilm):
        # filename
        self._filename = filename

        # create repository
        os.mkdir(self._filename)

        # biofilm
        self._biofilm = biofilm

        # tables
        self._coco_history, self._block_history, self._gene_instance = \
            CocoStep(biofilm.medium), BlockStep(biofilm.medium), GeneStep(biofilm.medium)

        # universe
        self._universe_step = UniverseStep(biofilm.medium)

    def get_filename(self):
        # filename
        return self._filename

    def run(self):
        # biofilm
        self._biofilm.step()

        # collect step
        self._coco_history.parse_step()
        self._block_history.parse_step()

    def done(self):
        # remove tree
        shutil.rmtree(self._filename)

        # create repository
        os.mkdir(self._filename)

        # store genes
        self._gene_instance.parse_step()

        # store tables
        self._coco_history.get_table().to_csv(self._filename + "/coco.csv", index=False)
        self._block_history.get_table().to_csv(self._filename + "/block.csv", index=False)
        self._gene_instance.get_table().to_csv(self._filename + "/gene.csv", index=False)

        sys.setrecursionlimit(10000)
        pickle.dump(self._universe_step, open(self._filename + "/universe.pkl", "wb"))


class StoredBlock(Cell):
    def __init__(self, **kwargs):
        # initial stack of substrates on each cell
        self._container = dict()

        # flux
        self._flux = 0

        # initialize cell
        super().__init__(limit=math.inf, **kwargs)

    def get_substrate(self, name):
        # return substrate amount
        if name in self._container:
            return self._container[name]

        # no substrate
        return 0

    def get_moles(self):
        # get substrates
        substrates = self.get_container()
        total = sum([substrates[s] for s in substrates])

        # return total amount of moles
        return total

    def get_container(self):
        # substrate container
        return self._container

    def add_flux(self, flux):
        self._flux += flux

    def get_gene_flux(self):
        return self._flux

    def get_substrate(self, name):
        # return substrate amount
        if name in self._container:
            return self._container[name]

        # no substrate
        return 0

    def put_substrate(self, name, amount):
        if name not in self._container:
            # new substrate
            self._container[name] = amount
        else:
            # accumulate
            self._container[name] += amount

        assert self._container[name] >= 0

    def put_substrates(self, substrates):
        for substrate in substrates:
            self.put_substrate(substrate, substrates[substrate])


class StoredCoco(Particle):
    def __init__(self, row, **kwargs):
        # step count
        self._count = 0

        # row
        self._row = row

        # states
        self._unbalanced, self._starving, self._scarce, self._planktonic = False, False, False, False

        # position
        position = [row.iloc[self._count]["x"], row.iloc[self._count]["y"]]

        # container
        self._container = None

        # container
        self._genes = None

        # update state
        self.update_state()

        # radius
        self.radius = self.get_radius()
        self.color = self.get_color()

        # initialize super
        super().__init__(position=position, **kwargs)

    def is_unbalanced(self):
        # stress state
        return self._unbalanced

    def is_starved(self):
        # stress state
        return self._starving

    def is_scarce(self):
        # stress state
        return self._scarce

    def is_planktonic(self):
        # stress state
        return self._planktonic

    def is_critical(self):
        # critical state
        return self.is_unbalanced() and self.is_starved() and self.is_scarce()

    def get_color(self):
        # normal color
        normal = 0.30

        # starving or scarce
        if self._starving or self._scarce:
            normal = 0.60

        # stress state
        if self._unbalanced:
            normal = 0.85

        # color
        return 1.0 if (self._unbalanced and self._starving and self._scarce) else normal

    def get_radius(self):
        return 0.003 * (self.get_moles() / CHAMBER_MOLES)

    def get_substrate(self, name):
        # return substrate amount
        if name in self._container:
            return self._container[name]

        # no substrate
        return 0

    def get_moles(self):
        # get substrates
        substrates = self.get_container()
        total = sum([substrates[s] for s in substrates])

        # return total amount of moles
        return total

    def get_container(self):
        # substrate container
        return self._container

    def update_state(self):
        # update state
        self._container = json.loads(self._row.iloc[self._count]["container"])
        self._genes = json.loads(self._row.iloc[self._count]["genes"])
        self._position = numpy.array([self._row.iloc[self._count]["x"], self._row.iloc[self._count]["y"]])

        self._unbalanced, self._starving, self._scarce, self._planktonic = \
            self._row.iloc[self._count]["unbalanced"], self._row.iloc[self._count]["starving"], \
            self._row.iloc[self._count]["scarce"], self._row.iloc[self._count]["planktonic"]

        # update color
        right, left = self.get_substrate("CNCNCNCN"),  self.get_substrate("NCNCNCNC")
        if right > left:
            self.color = 0.30
        else:
            self.color = 0.70

    def interact(self, signals):
        try:
            self.update_state()
        except IndexError:
            yield Gone()

        # one step
        self._count += 1


class StoredDish(Box):
    def __init__(self, filename, **kwargs):
        # filename
        self._filename = filename

        # get universe parameters
        self._universe = pickle.load(open(self._filename + "/universe.pkl", "rb"))

        # get history
        self._block_history = pandas.read_csv(self._filename + "/block.csv")

        # get cells
        self._max_time = self._block_history["time"].max()

        # cell index
        idx = self._block_history["idx"].max()
        idy = self._block_history["idy"].max()

        # initialize box
        super().__init__(grid=(idx + 1, idy + 1), cell=StoredBlock, bottom_bounds=self._universe.bottom_bounds,
                         upper_bounds=self._universe.upper_bounds, **kwargs)

        # initial cells
        cells = self._block_history[self._block_history["time"] == 1]

        # instance cells
        for cell in self.get_cells():
            container = json.loads(cells[(cells["idx"] == cell.index[0]) &
                                         (cells["idy"] == cell.index[1])]["container"].values[0])

            cell.put_substrates(container)

        # coconatas
        self._coconatas = dict()

        # get history
        self._coco_history = pandas.read_csv(self._filename + "/coco.csv", chunksize=5*10**7)

        # refresh cocos
        self.refresh_cocos()

    def refresh_cocos(self):
        if len(self._coconatas) <= 40:
            chunk = next(iter(self._coco_history))

            # instantiate
            for idx in chunk["idx"].unique():
                # get row
                row = chunk[chunk["idx"] == idx]

                # get born time
                born_time = row["time"].min()

                # store
                if born_time not in self._coconatas:
                    self._coconatas[born_time] = list()

                # append
                self._coconatas[born_time].append(row)

    def get_substrates(self, condition=None):
        # agents
        agents = self.get_agents(condition=condition)

        # substrates collection
        substrates = dict()

        # collect substrates
        for agent in agents:
            cell_substrates = {s: agent.get_substrate(s) for s in agent.get_container()}

            # collect substrates
            for each in cell_substrates:
                if each not in substrates:
                    substrates[each] = cell_substrates[each]
                else:
                    substrates[each] += cell_substrates[each]

        # get back species
        return substrates

    def get_substrate_mesh(self, substrate):
        # food and energy release
        matrix = numpy.zeros((self._grid[1], self._grid[0]), float)

        # food mesh
        for each in self.get_children():
            matrix[each.index[1], each.index[0]] = each.get_substrate(substrate)

        # return matrix
        return matrix

    def get_gene_flux_mesh(self):
        # food and energy release
        matrix = numpy.zeros((self._grid[1], self._grid[0]), float)

        # food mesh
        for each in self.get_children():
            matrix[each.index[1], each.index[0]] = each.get_gene_flux()

        # return matrix
        return matrix

    def get_coconatas(self):
        return self._coconatas

    def __call__(self, *args, **kwargs):
        # the coco enigma ||||||||||||||||||||||||||||||||||||||||||||||#
        coconatas = self.get_coconatas()                                #
        if (self.get_time() - 1) in coconatas and self.get_time() > 2:  #
            while len(coconatas[(self.get_time() - 1)]) > 0:            #
                row = coconatas[(self.get_time() - 1)].pop(0)           #
                self.put(StoredCoco(row=row))                           #
        # ==============================================================#

        # initial cells
        cells = self._block_history[self._block_history["time"] == self.get_time()]

        # instance cells
        if len(cells) > 0:
            for cell in self.get_cells():
                container = json.loads(cells[(cells["idx"] == cell.index[0]) &
                                             (cells["idy"] == cell.index[1])]["container"].values[0])

                cell.put_substrates(container)

                flux = cells[(cells["idx"] == cell.index[0]) &
                             (cells["idy"] == cell.index[1])]["flux"].values[0]

                cell.add_flux(flux)

        if self.get_time() - 1 in self._coconatas and len(self._coconatas[(self.get_time() - 1)]) == 0:
            del self._coconatas[(self.get_time() - 1)]

        # refresh cocos
        self.refresh_cocos()
