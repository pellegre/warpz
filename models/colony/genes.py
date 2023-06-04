from models.colony.common import *

from enum import IntEnum

import numpy
import bisect
import copy


# ===========
#
# drive
#
# ===========

class Drive:
    def init(self, coconata):
        # setup rewards
        pass

    def get_reward(self, coconata):
        # get drive reward
        raise RuntimeError("get_reward not implemented")


# ===========
#
# control unit
#
# ===========

class Decision(IntEnum):
    IDLE = 0
    MOVE = 1


class Action(IntEnum):
    POLYMERIZATION = 0
    CLEAVAGE = 1
    OUTWARD = 2
    INWARD = 3
    BUILD = 4
    BREAK = 5
    NONE = 6


# action names
ACTION_NAMES = {Action.POLYMERIZATION: "ANABOLIC",
                Action.CLEAVAGE: "CATABOLIC",
                Action.INWARD: "IN",
                Action.OUTWARD: "OUT",
                Action.BUILD: "BUILD",
                Action.BREAK: "BREAK"}


class Gene:
    # crossover parameters
    REPLICATION_RATE = 0.05

    # learning parameters
    EPSILON = 0.01
    GAMMA = 0.10
    ALPHA = 0.75

    # moles captured
    MOLES = 5

    # mutation rate
    MUTATION = 0.10

    def __init__(self, substrate, action: Action, states_size=16, **kwargs):
        # unit name
        self._substrate = substrate

        # action to take
        self._action = action

        # Q learning matrix
        self._states_size = states_size
        self._q_matrix = numpy.random.uniform(0, 1, (self._states_size, Gene.MOLES + 1))

        # epsilon (switching from greedy to random behavior)
        self._epsilon = {s: 0.50 for s in range(0, self._states_size)}

        # Q parameters
        self._gamma = random_state.uniform(0.25, Gene.ALPHA)
        self._alpha = random_state.uniform(0.00, Gene.GAMMA)

        # mutation rate
        self._mutation = random_state.uniform(0, Gene.MUTATION)

        # replication rate
        self._replication_rate = random_state.uniform(0, Gene.REPLICATION_RATE)

        # moles
        self._moles = Gene.MOLES

        # track state an reward tokens
        self._state = None

        # decision rewards
        self._decision_reward = {Decision.MOVE: 0, Decision.IDLE: 0}

    def print(self):
        print("[+] gene", self.get_name())
        print("[+]    decision reward", self._decision_reward)
        print("[+]    action", self._action)
        print("[+]    Q matrix")
        print(self._q_matrix)

    @staticmethod
    def get_state(coconata):
        # update coconata state
        coconata.state()

        # get coconata state
        return (0 if coconata.is_unbalanced() else 1) + 2 * (0 if coconata.is_starved() else 1) + \
               4 * (0 if coconata.is_scarce() else 1) + 8 * (0 if coconata.is_pressurized() else 1)

    def init(self, coconata):
        # initialize state
        self._state = self.get_state(coconata)

    def get_replication_rate(self):
        # gene replication rate
        return self._replication_rate

    def mutate(self):
        if random_state.uniform(0, 1) < self._mutation:
            self._replication_rate = random_state.normal(self._replication_rate, (self._replication_rate / 10))

        if random_state.uniform(0, 1) < self._mutation:
            self._alpha = random_state.normal(self._alpha, (self._alpha / 10))

        if random_state.uniform(0, 1) < self._mutation:
            self._gamma = random_state.normal(self._gamma, (self._gamma / 10))

        if random_state.uniform(0, 1) < self._mutation:
            self._mutation = random_state.normal(self._mutation, (self._mutation / 10))

    def get_substrate(self):
        # get unit name
        return self._substrate

    def get_name(self):
        # get name of the unit
        return str(ACTION_NAMES[self._action] + "@" + (self._substrate[0] + "-" + self._substrate[1]
                                                       if isinstance(self._substrate, tuple) else self._substrate))

    @staticmethod
    def get_reward(coconata):
        # get back reward
        return coconata.get_substrate("R")

    def set_moles(self, moles):
        # get moles for action
        self._moles = moles

    def get_moles(self):
        # get moles for action
        return self._moles

    def get_action(self):
        # translate unit decision
        if self._moles > 0:
            return self._action

        # no action
        return Action.NONE

    def reward_action(self, coconata):
        # store previous state
        previous = self._state

        # scan new state and reward
        current, reward = self.get_state(coconata), self.get_reward(coconata)

        # current decision
        a = self._moles

        # accumulate reward
        self._decision_reward[Decision.MOVE if self._moles > 0 else Decision.IDLE] += reward

        # update Q matrix
        self._q_matrix[previous, a] = self._q_matrix[previous, a] + self._alpha * \
            (reward + self._gamma * numpy.max(self._q_matrix[current, :] - self._q_matrix[previous, a]))

        # update state
        self._state = current

        # update epsilon
        self._epsilon[current] -= Gene.EPSILON
        self._epsilon[current] = max(Gene.EPSILON, self._epsilon[current])

    def choose_action(self, coconata):
        # scan new state
        current = self.get_state(coconata)

        # sample and press button
        if random_state.uniform(0, 1) < self._epsilon[current]:
            # random action
            action_taken = random_state.randint(0, Gene.MOLES + 1)

        else:
            # get state row
            row = self._q_matrix[current, :].flatten()

            # substrates fraction
            accumulated = numpy.add.accumulate(row) / sum(row)

            # sample random number
            sample = random_state.uniform(0, 1)

            # choose action
            action_taken = bisect.bisect(accumulated, sample)

        # update action
        self._moles = action_taken if action_taken <= Gene.MOLES else random_state.randint(0, Gene.MOLES + 1)

    def mutate_molecule(self, substrate):
        # get reactants
        one = list(substrate)

        # mutate one reactant
        if random_state.uniform(0, 1) < self._mutation:
            one[random_state.randint(0, len(one))] = ["N", "C"][random_state.randint(0, 2)]

        # random insert
        if random_state.uniform(0, 1) < self._mutation:
            one.append(["N", "C"][random_state.randint(0, 2)])

        # random delete
        if len(one) > 1 and random_state.uniform(0, 1) < self._mutation:
            del one[random_state.randint(0, 2)]

        # recover
        return ''.join(one)

    @staticmethod
    def get_random_molecule(size):
        # get elements
        c = random_state.randint(0, size)
        n = size - c

        # shuffle pump
        molecule = ["C"] * c + ["N"] * n
        numpy.random.shuffle(molecule)

        # mutant pump
        mutant = ''.join(molecule)

        # check for proteo pumps
        while len(mutant) % 2 == 0 and mutant.count("CN") == len(mutant) // 2 and \
                mutant.count("NC") == len(mutant) // 2:
            # shuffle it
            numpy.random.shuffle(molecule)
            mutant = ''.join(molecule)

            # update pump
            molecule = ["C"] * c + ["N"] * n

        return mutant

    def mutate_substrate(self, random=False):
        if isinstance(self._substrate, str):
            if random:
                return self.get_random_molecule(random_state.randint(4, 7))
            return self.mutate_molecule(self._substrate)
        elif isinstance(self._substrate, tuple):
            if random_state.uniform(0, 1) < 0.50:
                return self.mutate_molecule(self._substrate[0]), self._substrate[1]
            else:
                return self._substrate[0], self.mutate_molecule(self._substrate[1])

    def replicate(self, random=False):
        # decision and substrate
        decision, substrate = self._action, self._substrate

        if decision is Action.BUILD or decision is Action.BREAK:
            # mutate decision
            if random_state.uniform(0, 1) < self._mutation:
                decision = Action.BUILD if random_state.uniform(0, 1) < 0.50 else Action.BREAK
        else:
            # mutate
            if decision is Action.OUTWARD or decision is Action.INWARD:
                # mutate decision
                decision = Action.OUTWARD

            if decision is Action.POLYMERIZATION or decision is Action.CLEAVAGE:
                # mutate decision
                if random_state.uniform(0, 1) < self._mutation:
                    decision = Action.POLYMERIZATION if random_state.uniform(0, 1) < 0.50 else Action.CLEAVAGE

            # and substrate
            substrate = self.mutate_substrate(random)

        # create gene
        gene = Gene(substrate=substrate, action=decision)
        gene.mutate()

        # get back gene
        return gene


# ===========
#
# biological drives
#
# ===========

class PleasureDrive(Drive):
    def __init__(self, **kwargs):
        # initialize drive
        super(PleasureDrive, self).__init__()

    def get_reward(self, coconata):
        # state
        return int(coconata.get_substrate("R"))


class PressureDrive(Drive):
    def __init__(self, **kwargs):
        # initialize drive
        super(PressureDrive, self).__init__()

    def get_reward(self, coconata):
        # state
        return int(min(0, HIGH_PRESSURE_MOLES - coconata.get_moles()))


class AnabolicDrive(Drive):
    def __init__(self, substrate, **kwargs):
        # anabolic substrate
        self._substrate = substrate

        # initialize drive
        super(AnabolicDrive, self).__init__()

    def get_substrate(self):
        # return anabolic substrate
        return self._substrate

    def get_reward(self, coconata):
        # state
        return coconata.get_substrate(self._substrate)


class CatabolicDrive(Drive):
    def __init__(self, **kwargs):
        # initialize drive
        super(CatabolicDrive, self).__init__()

    def get_reward(self, coconata):
        # get substrate container
        container = coconata.get_container()

        # scan new state
        chemistry = coconata.get_dish().get_chemistry()
        price, token = chemistry.get_token_price(), chemistry.get_token()

        # state
        return (2 * coconata.get_substrate(token) +
                          sum([chemistry.get_bond_energy(s) * container[s] for s in container])) * price

# ===========
#
# gene
#
# ===========


class Chromosome:
    def __init__(self, pumps, reactions):
        # greediness
        self._greediness = random_state.randint(5, 10)

        # mutation rate
        self._mutation = random_state.uniform(0, Gene.MUTATION)

        # pump coco units
        self._pump_units = pumps

        # reaction coco units
        self._reaction_units = reactions

        # mother genes
        self._plasmid_pumps, self._plasmid_reactions = list(), list()

    def get_genes(self):
        # get genes
        return {gene.get_name() for gene in self.get_pumps()}.union({gene.get_name() for gene in self.get_reactions()})

    def mutate(self):
        if random_state.uniform(0, 1) < self._mutation:
            self._greediness = random_state.randint(5, 10)

        # mutate pumps
        for pump in list(self._pump_units):
            # check moles
            if random_state.uniform(0, 1) < self._mutation:
                self._pump_units.remove(pump)
            else:
                # mutate pump
                pump.mutate()

        # mutate pumps
        for reaction in list(self._reaction_units):
            # check moles
            if random_state.uniform(0, 1) < self._mutation:
                self._reaction_units.remove(reaction)
            else:
                # mutate reaction
                reaction.mutate()

        if random_state.uniform(0, 1) < self._mutation:
            self._mutation = random_state.normal(self._mutation, (self._mutation / 10) * self._mutation)

    def get_greediness(self):
        return self._greediness

    # ======
    # coconata pumps and reactions management
    #
    def get_pumps(self):
        # pump matrix
        return self._pump_units + self._plasmid_pumps

    def insert_pump(self, pump):
        # insert pump
        if len(self._plasmid_pumps) < PLASMID_GENES:
            # append pump
            self._plasmid_pumps.append(pump)
        else:
            # replace pump
            idx = random_state.randint(0, len(self._plasmid_pumps))
            self._plasmid_pumps[idx] = pump

    def get_reactions(self):
        # reaction matrix
        return self._reaction_units + self._plasmid_reactions

    def insert_reaction(self, reaction):
        # insert pump
        if len(self._plasmid_reactions) < PLASMID_GENES:
            # put reaction
            self._plasmid_reactions.append(reaction)
        else:
            # pick up reaction reaction
            idx = random_state.randint(0, len(self._plasmid_reactions))
            self._plasmid_reactions[idx] = reaction


class Mother(Chromosome):
    def __init__(self, target, anabolic, catabolic, pressure, matrix=True):
        # proteo drive
        self._replication_drive = AnabolicDrive(substrate=target)

        # energetic drive
        self._catabolic_drive = CatabolicDrive()

        # pressure drive
        self._pressure_drive = PressureDrive()

        # pressure drive
        self._pleasure_drive = PleasureDrive()

        # reaction units
        reactions = list()

        # anabolic reactions
        for reactants in anabolic + catabolic:
            reaction = Gene(substrate=reactants, action=Action.POLYMERIZATION)
            reactions.append(reaction)

        # catabolic reactions
        for reactants in catabolic:
            reaction = Gene(substrate=reactants, action=Action.CLEAVAGE)
            reactions.append(reaction)

        # setup units
        pumps = list()

        # sample pumps
        for substrate in pressure:
            # sample inward
            inward = Gene(substrate=substrate, action=Action.INWARD)
            pumps.append(inward)

            # sample outward
            if self.is_c_molecule(  substrate):
                outward = Gene(substrate=substrate, action=Action.OUTWARD)
                pumps.append(outward)

        # matrix builder
        if matrix:
            # matrix builder pump
            build = Gene(substrate="E", action=Action.BUILD)
            pumps.append(build)

            # matrix builder pump
            dismantle = Gene(substrate="E", action=Action.BREAK)
            pumps.append(dismantle)

        # drives
        self._drives = [self._replication_drive, self._catabolic_drive, self._pressure_drive]

        # initialize mother gene
        super(Mother, self).__init__(pumps=pumps, reactions=reactions)

    @staticmethod
    def is_c_molecule(substrate):
        return all([c == "C" for c in substrate])

    def get_anabolic_drive(self):
        # get anabolic drive molecule
        return self._replication_drive.get_substrate()

    def get_pleasure_drive(self):
        # get back pleasure
        return self._pleasure_drive

    def get_drives(self):
        # get back chromosome drives
        return self._drives


class RightSpecie(Mother):
    def __init__(self, **kwargs):
        # anabolic target molecule
        target = "CNCNCNCN"

        # reaction units
        anabolic = [("C", "N"), ("CN", "CN"), ("CNCN", "CN"), ("CNCNCN", "CN"), ("CNCN", "CNCN")]
        catabolic = [("C", "C"), ("C", "CC"), ("CC", "CC"), ("C", "CCC"), ("CC", "CC")]

        # pump space
        pressure = ["C", "N", "CN", "CC", "CNCN", "CNCNCN", "CNCNCNCN",  "CCC", "CCCC"]

        super().__init__(target=target, anabolic=anabolic, catabolic=catabolic, pressure=pressure, **kwargs)


class LeftSpecie(Mother):
    def __init__(self, **kwargs):
        # anabolic target molecule
        target = "NCNCNCNC"

        # reaction units
        anabolic = [("N", "C"), ("NC", "NC"), ("NCNC", "NC"), ("NCNCNC", "NC"), ("NCNC", "NCNC")]
        catabolic = [("C", "C"), ("C", "CC"), ("CC", "CC"), ("C", "CCC"), ("CC", "CC")]

        # pump space
        pressure = ["C", "N", "NC", "CC", "NCNC", "NCNCNC", "NCNCNCNC", "CCC", "CCCC"]

        super().__init__(target=target, anabolic=anabolic, catabolic=catabolic, pressure=pressure, **kwargs)
