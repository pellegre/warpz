from warpz.space.box import *

from models.colony.chemistry import *
from models.colony.genes import *

from enum import IntEnum

import math
import numpy
import time
import copy

# floating point epsilon
EPSILON = 10E-07


# ===========
#
# dish block
#
# ===========


class Block(Cell):
    def __init__(self, **kwargs):
        # initial stack of substrates on each cell
        self._container = dict()

        # universe reference
        universe = kwargs["universe"]

        # reactor
        self._reactor = universe.get_chemistry().get_reactor(universe=universe)

        # gene flux
        self._gene_flux = 0

        # signals queue
        self._signal_queue = list()

        # initialize cell
        super().__init__(limit=math.inf, **kwargs)

    def setup(self, chemical: Chemical):
        # clean up substrates container
        for substrate in chemical.get_elements():
            self._container[substrate] = 0

    def add_signal(self, signal):
        # queue signal
        self._signal_queue.append(signal)

    def get_signal(self):
        # get signal in the queue
        if len(self._signal_queue) > 0:
            return self._signal_queue.pop()

        # no signal right now
        return None

    # ======
    # chemistry management
    #
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

    def get_element(self, name):
        # element amount
        amount = 0

        # return element amount
        for each in self._container:
            if name == each:
                amount += self._container[each]
            else:
                amount += int(self._container[each] * each.count(name))

        # no substrate
        return amount

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

    # ======
    # reaction handling
    #
    def polymerization(self, a, b, amount):
        # yield substrate through the reaction
        return self._reactor.polymerization(self, a, b, amount)

    def cleave(self, a, b, amount=1):
        # cleave substrate through the reactor
        return self._reactor.cleave(self, a, b, amount)

    def breakage(self, substrate, amount=1):
        # break substrate through the reactor
        return self._reactor.breakage(self, substrate, amount)

    def thermal(self):
        # thermal reaction
        return self._reactor.thermal(self)

    def gene_flux(self):
        # count gene flux
        self._gene_flux += 1

    def get_gene_flux(self):
        # get back gene flux
        return self._gene_flux / (self.get_time() + 1)

    # ======
    # interaction
    #
    def interact(self, signals):
        # thermal reaction
        self.thermal()


# ===========
#
# colony dish
#
# ===========


class Dish(Box):
    def __init__(self, chemical: Chemical, grid=10, **kwargs):
        # chemical elements
        self._chemical = chemical

        # initialize box
        super().__init__(grid=grid, cell=Block, **kwargs)

        # motility
        delta = (self.get_upper_bounds() - self.get_bottom_bounds()) / grid
        self._motility = math.sqrt(delta[0] ** 2 + delta[1] ** 2) / 20

        # substrates container
        self._container = {e: chemical.get_total_amount(e) for e in chemical.get_elements()}

        # setup substrate cell distribution
        self._distribution = {s: [cell for cell in self.get_cells() if s in chemical.distribution(cell)]
                              for s in chemical.get_elements()}

        # setup containers in cell
        for each in self.get_cells():
            # setup chemicals
            each.setup(chemical)

        # distribute substrates
        self.distribute()

        # initial time
        self._tic, self._toc = time.process_time(), time.process_time()

        # cost function
        self._clock_time, self._cost = 0, 1

    def get_motility(self):
        # get universe motility
        return self._motility

    # ======
    # chemistry management
    #
    def get_chemistry(self):
        # return chemistry space
        return self._chemical

    def get_container(self):
        # substrate container
        return self._container

    def get_substrate(self, substrate):
        # return substrate amount
        if substrate in self._container:
            return self._container[substrate]

        # no substrate
        return 0

    def get_element(self, name):
        # element amount
        amount = 0

        # return element amount
        for each in self._container:
            if name == each:
                amount += self._container[each]
            else:
                amount += int(self._container[each] * each.count(name))

        # no substrate
        return amount

    def put_substrate(self, substrate, amount):
        if substrate not in self._container:
            # new substrate
            self._container[substrate] = amount
        else:
            # accumulate
            self._container[substrate] += amount

    def put_substrates(self, substrates):
        for substrate in substrates:
            self.put_substrate(substrate, substrates[substrate])

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

    # ======
    # energy statistics
    #
    def get_dissipated_energy(self):
        # get dissipated energy
        return self._chemical.get_token_price() * self._container[self._chemical.get_token()]

    def get_free_energy(self, condition=None):
        # agents
        agents = self.get_agents(condition=condition)

        # get energy in free tokens
        return sum([self._chemical.get_token_price() * agent.get_substrate(self._chemical.get_token())
                    for agent in agents])

    def get_bond_energy(self, condition=None):
        # agents
        agents = self.get_agents(condition=condition)

        # get energy in bonds
        bond_energy = 0

        # bond energy per cell`
        for agent in agents:
            bond_energy += sum([agent.get_substrate(s) * self._chemical.get_bond_energy(s)
                                for s in agent.get_container()])

        # total energy in bonds
        return bond_energy

    # ======
    # substrates statistics
    #
    def get_substrates(self, condition=None):
        # agents
        agents = self.get_agents(condition=condition)

        # substrates collection
        substrates = dict()

        # collect substrates
        for agent in agents:
            cell_substrates = {s: agent.get_substrate(s) for s in agent.get_container()
                               if s is not self._chemical.get_token()}

            # collect substrates
            for each in cell_substrates:
                if each not in substrates:
                    substrates[each] = cell_substrates[each]
                else:
                    substrates[each] += cell_substrates[each]

        # get back species
        return substrates

    def get_elements(self, condition=None):
        # agents
        agents = self.get_agents(condition=condition)

        # substrates collection
        elements = dict()

        # collect substrates
        for agent in agents:
            cell_elements = {s: agent.get_element(s) for s in self._chemical.get_elements()
                             if s is not self._chemical.get_token()}

            # collect substrates
            for each in cell_elements:
                if each not in elements:
                    elements[each] = cell_elements[each]
                else:
                    elements[each] += cell_elements[each]

        # get back elements
        return elements

    # ======
    # simulation cost management
    #
    def get_cost(self):
        # simulation cost
        return self._cost

    def update_cost(self):
        # initial time
        self._toc = time.process_time()

        # current delta
        delta = self._toc - self._tic

        # update clock delta average
        self._clock_time += delta

        # get average
        clock_delta = self._clock_time / (self.get_time() + 1)

        # update last clock
        self._tic = self._toc

        # update cost
        self._cost = int(delta / clock_delta) + 1

        # set token price
        # self._chemical.set_token_price(1 / self._cost)

    # ======
    # substrate distribution, diffusion and reaction
    #
    def distribute(self):
        # distribute substrates
        for substrate in self._distribution:
            # get cells
            cells = self._distribution[substrate]

            if self.get_substrate(substrate) > 0 and len(cells) > 0:
                # total amount
                total = self.get_substrate(substrate)

                # substrate amount for each cell
                each = int(total / len(cells))

                # distribute energy
                for cell in cells:
                    if substrate is self._chemical.get_token():
                        # get token price
                        price = self._chemical.get_token_price()

                        # get bonds
                        bonds = 1

                        # injected substrate
                        s = ''.join(["C"] * (bonds + 1))

                        # minimal amount per cell
                        minimal = min((each * price) / self._chemical.get_bond_energy(s),
                                      self.get_substrate(s) // (bonds + 1))
                        amount = (minimal * self._chemical.get_bond_energy(s)) / price

                        # token distribution
                        self._container[substrate] -= amount
                        cell.put_substrate(substrate, minimal)
                    else:
                        # minimal amount per cell
                        amount = min(each, self.get_substrate(substrate))

                        # token distribution
                        self._container[substrate] -= amount
                        cell.put_substrate(substrate, amount)

            rest = self._container[substrate]
            if rest > 0:
                self._container[substrate] -= rest
                cells[random_state.randint(0, len(cells))].put_substrate(substrate, rest)

    def diffuse(self):
        # distribute substrates
        for cell in self.get_cells():
            for substrate in filter(lambda s: cell.get_substrate(s) > 0, cell.get_container()):
                # get neighbors
                neighbors = self.get_periodic_neighbors(cell)
                numpy.random.shuffle(neighbors)

                # sample diffusive substrates
                diffusion = self._chemical.get_diffusion(substrate, medium=cell) / len(neighbors)
                diffuse = numpy.random.binomial(cell.get_substrate(substrate), diffusion, len(neighbors))

                # distribute substrate
                for i, each in enumerate(neighbors):
                    # balance substrate
                    amount = min(cell.get_substrate(substrate), diffuse[i])

                    # put substrate
                    cell.put_substrate(substrate, -amount)
                    each.put_substrate(substrate, amount)

                    # sanity check
                    assert cell.get_substrate(substrate) >= 0
                    assert each.get_substrate(substrate) >= 0

    def __call__(self, *args, **kwargs):
        # update simulation cost
        self.update_cost()

        # diffuse
        self.diffuse()

        # distribute
        self.distribute()


# ===========
# actions
#

class Diffuse(IntEnum):
    INWARD = 0
    OUTWARD = 1


# ===========
#
# coconata
#
# ===========

class Coconata(Particle):
    def __init__(self, dish: Dish, container: dict, chromosome: Mother, **kwargs):
        # initialize super
        super().__init__(**kwargs)

        # coconata gene
        self._chromosome = chromosome

        # substrate container
        self._container = container

        # instantiate reactor
        self._reactor = dish.get_chemistry().get_reactor(universe=dish)

        # store dish
        self._dish = dish

        # stress state
        self._unbalanced, self._starving, self._scarce, \
            self._planktonic, self._pressurized = False, False, False, False, False

        # gene tallies
        self._tallies = dict()

        # born agent
        self._born = True

    def get_dish(self):
        # get back dish
        return self._dish

    def get_tallies(self):
        # get gene tallies
        return self._tallies

    def state(self):
        # setup starving state
        self._scarce = True if self.get_substrate(self._chromosome.get_anabolic_drive()) < CHAMBER_MOLES // 6 \
            else False

        # get token price
        price = self._dish.get_chemistry().get_token_price()

        # calculate amount of energy
        container = self.get_container()
        energy = (self.get_substrate("Z") +
                  sum([self._dish.get_chemistry().get_bond_energy(s) * container[s] for s in container])) * price

        # starving state
        self._starving = True if energy < CHAMBER_MOLES // 6 else False

        # unbalanced state
        self._unbalanced = True if self.get_substrate("R") < CHAMBER_MOLES // 6 else False

        # setup planktonic state
        transition_matrix = MATRIX_MOLES / MATRIX_MOTILITY
        matrix = self._dish.get_cell(self.get_position()).get_substrate("E")

        # planktonic state
        self._planktonic = True if matrix < transition_matrix else False

        # pressurized state
        self._pressurized = True if self.get_moles() >= HIGH_PRESSURE_MOLES else False

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

    def is_pressurized(self):
        # stress state
        return self._pressurized

    def is_critical(self):
        # critical state
        return self.is_starved() or self.is_pressurized()

    def get_chromosome(self):
        # return chromosome
        return self._chromosome

    def just_born(self):
        if self._born:
            # update value
            self._born = False

            # just born
            return True

        # return value
        return self._born

    # ======
    # chemistry management
    #
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

    def get_element(self, name):
        # element amount
        amount = 0

        # return element amount
        for each in self._container:
            if name == each:
                amount += self._container[each]
            else:
                amount += int(self._container[each] * each.count(name))

        # no substrate
        return amount

    def put_substrate(self, name, amount):
        if name not in self._container:
            # new substrate
            self._container[name] = amount
        else:
            # accumulate
            self._container[name] += amount

    def put_substrates(self, substrates):
        for substrate in substrates:
            self.put_substrate(substrate, substrates[substrate])

    # ======
    # reaction handling
    #
    def polymerization(self, a, b, amount):
        # yield substrate through the reactor
        return self._reactor.polymerization(self, a, b, amount)

    def cleave(self, a, b, amount):
        # cleave substrate through the reactor
        return self._reactor.cleave(self, a, b, amount)

    def breakage(self, substrate, amount=1):
        # break substrate through the reactor
        return self._reactor.breakage(self, substrate, amount)

    def thermal(self):
        # thermal reaction
        return self._reactor.thermal(self)

    def sample_molecule(self, substrates):
        # get chemical
        chemical = self.get_universe().get_chemistry()

        # get substrates
        total = sum([substrates[s] for s in substrates if s is not chemical.get_token()])

        # substrates fraction
        fraction = {s: substrates[s] / total if total > 0 else 0 for s in substrates
                    if s is not chemical.get_token()}

        # sample reaction
        substrate, accumulated = list(), list()
        for i, each in enumerate(fraction):
            # append name
            substrate.append(each)

            # accumulate
            if i > 0:
                accumulated.append(fraction[each] + accumulated[i - 1])
            else:
                accumulated.append(fraction[each])

        # sample substrates
        i = bisect.bisect(accumulated, random_state.uniform(0, 1))

        # return substrate
        if i < len(substrate):
            return substrate[i]

        # no substrate
        return None

    # ======
    # diffusion
    #
    def diffuse_element(self, action, element, diffusion):
        # get cell
        cell = self.get_cell()

        # balance
        if action is Diffuse.OUTWARD:
            # sample outward flow
            outward = numpy.random.binomial(self.get_substrate(element), diffusion)

            # exchange flow
            self.put_substrate(element, -outward)
            cell.put_substrate(element, outward)

        elif action is Diffuse.INWARD:
            # sample outward flow
            inward = numpy.random.binomial(max(0, min(cell.get_substrate(element),
                                                      HIGH_PRESSURE_MOLES - self.get_moles())), diffusion)

            # exchange flow
            self.put_substrate(element, inward)
            cell.put_substrate(element, -inward)

    def diffuse_inward(self):
        # get tokens
        chemistry = self.get_universe().get_chemistry()
        token = chemistry.get_token()

        # current cell
        cell = self.get_cell()

        # absorbed tokens
        absorbed = min(self._chromosome.get_greediness(), cell.get_substrate(token))

        # get tokens
        self.put_substrate(token, absorbed)
        cell.put_substrate(token, -absorbed)

        # get container
        container = self.get_cell().get_container()

        # set elemental substrates
        elements = {s: container[s] for s in container if chemistry.can_diffuse_inward(s, self)}

        # diffuse in
        element = self.sample_molecule(elements)
        if element is not None:
            self.diffuse_element(Diffuse.INWARD, element, chemistry.get_membrane_diffusion(element))

    def diffuse_outward(self):
        # get tokens
        chemistry = self.get_universe().get_chemistry()

        # get container
        container = self.get_container()

        # set elemental substrates
        elements = {s: container[s] for s in container if (chemistry.can_diffuse_outward(s, self))}

        # diffuse out
        element = self.sample_molecule(elements)
        if element is not None:
            self.diffuse_element(Diffuse.OUTWARD, element, chemistry.get_membrane_diffusion(element))

    # ======
    # pump it up
    #
    def pump_substrate(self, action, substrate, moles=1):
        # hook matrix building
        if action is Action.BUILD or action is Action.BREAK:
            return self.pump_matrix(action, substrate, moles)

        # get cell
        cell = self.get_cell()

        # initial flow
        flow = 0

        # balance
        if action is Action.OUTWARD and moles > 0:
            # minimal pumped amount
            minimal = min(moles, self.get_substrate(substrate))

            # pump it
            if minimal > 0:
                # get token
                token, price = self.get_universe().get_chemistry().get_token(), \
                               self.get_universe().get_chemistry().get_token_price()

                # energy (one unit per molecule)
                energy = minimal

                # get amount
                tokens = min(energy / price, self.get_substrate(token))

                # calculate substrates
                flow = int(price * tokens)

                if flow > 0:
                    # token balance
                    if substrate is not FilmReactor.MATRIX_MOLECULE:
                        self.put_substrate(token, -tokens)
                        self.get_universe().put_substrate(token, tokens)

                    # exchange flow
                    self.put_substrate(substrate, -flow)
                    cell.put_substrate(substrate, flow)

        elif action is Action.INWARD and moles > 0:
            # minimal pumped amount
            minimal = min(moles, self.get_cell().get_substrate(substrate))

            # pump it
            if minimal > 0:
                # get token
                token, price = self.get_universe().get_chemistry().get_token(), \
                               self.get_universe().get_chemistry().get_token_price()

                # energy (one unit per molecule)
                energy = minimal

                # get amount
                tokens = min(energy / price, self.get_substrate(token))

                # calculate substrates
                flow = int(price * tokens)

                if flow > 0:
                    # token balance
                    if substrate is not FilmReactor.MATRIX_MOLECULE:
                        self.put_substrate(token, -tokens)
                        self.get_universe().put_substrate(token, tokens)

                    # exchange flow
                    self.put_substrate(substrate, flow)
                    cell.put_substrate(substrate, -flow)

        # get back amount
        return flow

    # ======
    # build matrix
    #
    def pump_matrix(self, action, substrate, moles=1):
        # pump matrix flow
        flow = 0

        if action is Action.BUILD and moles > 0:
            # matrix content
            matrix = self.get_cell().get_substrate("E")
            if matrix < MATRIX_MOLES:
                # put matrix molecule
                self.put_substrate(FilmReactor.MATRIX_MOLECULE, moles)

                # pump it outside
                flow = self.pump_substrate(Action.OUTWARD, substrate, moles)

        elif action is Action.BREAK and moles > 0:
            # pump it outside
            flow = self.pump_substrate(Action.INWARD, substrate, moles)

        # dissipate flow
        matrix = self.get_substrate(FilmReactor.MATRIX_MOLECULE)
        self.put_substrate(FilmReactor.MATRIX_MOLECULE, -matrix)

        # get back flow
        return flow

    # ======
    # react it up
    #
    def react_substrates(self, action, reactants, moles=1):
        # split tuple
        a, b = reactants

        # balance
        if action is Action.POLYMERIZATION and moles > 0:
            # polymerization reaction
            products = self.polymerization(a, b, amount=moles)
            self.put_substrates(products)

            # return products
            if len(products) > 0:
                return products[a + b]

        elif action is Action.CLEAVAGE and moles > 0:
            # breakage reaction
            products = self.cleave(a, b, amount=moles)
            self.put_substrates(products)

            if len(products) > 0:
                if a == b:
                    # same products
                    return products[a] // 2
                else:
                    # sanity
                    assert products[a] == products[b]

                    # return products
                    return products[a]

        # no reaction
        return 0

    # ======
    # pump
    #
    def pump(self):
        # pump out
        pumps = self._chromosome.get_pumps()
        numpy.random.shuffle(pumps)

        # select random pump
        for pump in pumps:
            # get action moles
            moles = pump.get_moles()

            # action
            action, name = pump.get_action(), pump.get_name()

            # accumulate
            if pump.get_name() not in self._tallies:
                self._tallies[name] = {"NONE": 0, "MOVE": 0}

            # pump substrate
            flow = self.pump_substrate(action, pump.get_substrate(), moles)

            # check moles flow
            assert flow <= moles

            # update moles flow
            pump.set_moles(flow)

            if flow > 0:
                # sanity
                assert action is not Action.NONE

                # accumulate action
                self._tallies[name]["MOVE"] += 1
            else:
                # sanity
                if moles == 0:
                    assert action is Action.NONE

                # accumulate action
                self._tallies[name]["NONE"] += 1

    # ======
    # react
    #
    def react(self):
        # polymer
        reactions = self._chromosome.get_reactions()
        numpy.random.shuffle(reactions)

        # perform random reaction
        for reaction in reactions:
            # get action moles
            moles = reaction.get_moles()

            # take action
            action, name = reaction.get_action(), reaction.get_name()

            # accumulate
            if reaction.get_name() not in self._tallies:
                self._tallies[name] = {"NONE": 0, "MOVE": 0}

            # perform reaction
            flow = self.react_substrates(reaction.get_action(), reaction.get_substrate(), moles)

            # check moles flow
            assert flow <= moles

            # update moles flow
            reaction.set_moles(flow)

            if flow > 0:
                # sanity
                assert action is not Action.NONE

                # accumulate reaction
                self._tallies[name]["MOVE"] += 1
            else:
                # sanity
                if moles == 0:
                    assert action is Action.NONE

                # accumulate reaction
                self._tallies[name]["NONE"] += 1

    # ======
    # sense and reward
    #
    def sense(self):
        # initialize sensing elements
        if self.just_born():
            # choose pump actions
            for pump in self._chromosome.get_pumps():
                # initialize state
                pump.init(self)

            # choose reaction actions
            for reaction in self._chromosome.get_reactions():
                # initialize state
                reaction.init(self)

            # initialize drives
            for drive in self._chromosome.get_drives():
                drive.init(self)

        # rewarding signal from drives
        reward = 0
        for drive in self._chromosome.get_drives():
            # accumulate rewards
            reward += drive.get_reward(self) // 10

        # maintain the reward level
        reward = (reward - self.get_substrate("R"))

        # current reward tokens
        if reward > 0:
            # release molecules
            self.put_substrate("R", reward)
        else:
            # dissipate reward molecules
            internal = min(self.get_substrate("R"), abs(reward))
            self.put_substrate("R", -internal)

        # reward pump units
        for pump in self._chromosome.get_pumps():
            pump.reward_action(self)

        # reward reaction units
        for reaction in self._chromosome.get_reactions():
            # collect reward
            reaction.reward_action(self)

        # choose pump actions
        for pump in self._chromosome.get_pumps():
            # get pump unit
            pump.choose_action(self)

        # choose reaction actions
        for reaction in self._chromosome.get_reactions():
            # get pump unit
            reaction.choose_action(self)

    # ======
    # react and pump
    #
    def react_and_pump(self):
        # catalytic reaction
        if self.get_time() % 2 == 0:
            # react
            self.react()

            # pump it
            self.pump()
        else:
            # pump it
            self.pump()

            # and react
            self.react()
