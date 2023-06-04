from models.colony.common import *

import bisect
import numpy


# max molecule size
MAX_BONDS = 4

# ===========
#
# chemicals
#
# ===========


class Substrate:
    def __init__(self, name, diffusion=0.05):
        # substrate name
        self._name = name

        # diffusion rate
        self._diffusion = diffusion

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        if not isinstance(other, Substrate):
            return False

        # compare name, how many
        return self._name == other._name

    def get_name(self):
        return self._name

    def get_diffusion(self):
        return self._diffusion


# ===========
#
# chemical reactor
#
# ===========


class Reactor:
    def __init__(self, universe, temperature=0.05, reaction_rate=0.50, max_bonds=MAX_BONDS):
        # universe reference
        self._universe = universe

        # reactor temperature
        self._temperature = temperature

        # reaction rate (versus break)
        self._reaction_rate = reaction_rate

        # max bonds
        self._max_bonds = max_bonds

    def is_reactive(self, medium, a, b):
        return True

    def can_break(self, a):
        return len(a) > 1

    def cleave(self, medium, a, b, amount):
        # chemical elements definition
        chemical = self._universe.get_chemistry()

        # minimal broken amount
        minimal = min(amount, medium.get_substrate(a + b))

        # break substrate
        if self.can_break(a + b) and minimal > 0:
            # count each element
            if a == b:
                count = {a: 2 * minimal}
            else:
                count = {a: minimal, b: minimal}

            # energy token and price
            token, price = chemical.get_token(), chemical.get_token_price()

            # calculate tokens
            tokens = chemical.get_bond_energy(a[-1] + b[0]) / price

            # take out substrate from the medium
            medium.put_substrate(a + b, -minimal)

            # make reaction
            return count | {token: minimal * tokens}

        return dict()

    def breakage(self, medium, substrate, amount):
        # chemical elements definition
        chemical = self._universe.get_chemistry()

        # minimal broken amount
        minimal = min(amount, medium.get_substrate(substrate))

        # break substrate
        if self.can_break(substrate) and minimal > 0:
            if substrate in chemical.get_elements():
                # element dissipation
                medium.put_substrate(substrate, -minimal)

                # nothing to return
                return dict()
            else:
                # count each element
                count = {c: substrate.count(c) * minimal for c in set(substrate)}

                # energy token and price
                token, price = chemical.get_token(), chemical.get_token_price()

                # calculate tokens
                tokens = chemical.get_bond_energy(substrate) / price

                # take out substrate from the medium
                medium.put_substrate(substrate, -minimal)

                # make reaction
                return count | {token: minimal * tokens}

        return dict()

    def polymerization(self, medium, a, b, amount):
        if self.is_reactive(medium, a, b):
            # concatenate substrate
            substrate = a + b

            # chemical elements definition
            chemical = self._universe.get_chemistry()

            # energy token and price
            token, price = chemical.get_token(), chemical.get_token_price()

            # get polymerized
            if a != b:
                polymerized = min(amount, min(medium.get_substrate(a), medium.get_substrate(b)))
            else:
                polymerized = min(amount, medium.get_substrate(a) // 2)

            # check if there are enough reactants
            if polymerized > 0:
                # energy reaction
                energy_reaction = chemical.get_reaction_energy(a[-1] + b[0])

                # check if there is enough energy
                energy = polymerized * energy_reaction

                # get amount
                tokens = min(energy / price, medium.get_substrate(token))

                # calculate substrates
                substrates = int(price * tokens)

                if substrates > 0:
                    # perform reaction
                    medium.put_substrate(a, -substrates)
                    medium.put_substrate(b, -substrates)

                    # use energy
                    medium.put_substrate(token, -tokens)

                    # put it back to the universe
                    bond_energy = tokens * price - substrates * chemical.get_bond_energy(a[-1] + b[0])
                    self._universe.put_substrate(token, bond_energy / price)

                    # return number of elements
                    return {substrate: substrates}

        # no reaction
        return dict()

    def thermal(self, medium):
        # chemical elements definition
        chemical = self._universe.get_chemistry()

        # thermal sampling
        thermal = random_state.uniform(0, 1)

        # thermal reaction
        if thermal < self._temperature:
            # get substrates
            substrates = medium.get_container()
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
            i, j = bisect.bisect(accumulated, random_state.uniform(0, 1)), \
                   bisect.bisect(accumulated, random_state.uniform(0, 1))

            # perform reaction
            if i < len(substrate) and j < len(substrate):
                # sample reaction
                reaction = random_state.uniform(0, 1)

                # reaction rate
                if reaction > self._reaction_rate:
                    # check max molecule size
                    y = self.polymerization(medium, substrate[i], substrate[j], 1)

                    # put back result
                    if len(y) > 0:
                        medium.put_substrates(y)

                else:
                    # sample reaction
                    cleave = random_state.uniform(0, 1)

                    if cleave < self._reaction_rate:
                        # break one
                        y = self.cleave(medium, substrate[i], substrate[j], 1)

                        # put back result
                        if len(y) > 0:
                            medium.put_substrates(y)

                    else:
                        # break one
                        x, y = self.breakage(medium, substrate[i], 1), self.breakage(medium, substrate[j], 1)

                        # put back result
                        if len(y) > 0:
                            medium.put_substrates(y)

                        # and for
                        if len(x) > 0:
                            medium.put_substrates(x)


class Chemical:
    def __init__(self):
        # substrate elements
        self._elements = dict()

        # substrates definition
        self._container = dict()

        # substrate diffusion rate
        self._diffusion = dict()

        # energy token
        self._token = None

        # initial price
        self._price = 1.00

    def get_reactor(self, **kwargs):
        # craft reactor
        return Reactor(**kwargs)

    def set_token(self, name):
        self._token = name

    def get_token(self):
        return self._token

    def get_token_price(self):
        return self._price

    def get_elements(self):
        # get elements
        return self._elements

    def get_total_amount(self, name):
        # get total amount of the element
        return self._container[name]

    def get_diffusion(self, name: str, medium=None):
        # count each element
        count = {c: name.count(c) for c in set(name)}

        # get diffusion
        diffusion = 0
        for each in count:
            diffusion += self._diffusion[each] / count[each]

        # averaging diffusion
        return diffusion / len(count)

    def get_membrane_diffusion(self, name: str):
        return self.get_diffusion(name)

    def can_diffuse_inward(self, name, medium):
        if name in self._elements:
            return True

        return False

    def can_diffuse_outward(self, name, medium):
        if name in self._elements and name is not self.get_token():
            return True

        return False

    @staticmethod
    def get_reaction_energy(substrate):
        return len(substrate) - 1

    @staticmethod
    def get_bond_energy(substrate):
        return 0

    def setup(self, substrate, initial):
        # add element
        self._elements[substrate.get_name()] = substrate

        # initial amount
        self._container[substrate.get_name()] = initial

        # set diffusion
        self._diffusion[substrate.get_name()] = substrate.get_diffusion()

    def distribution(self, cell):
        # get dish
        dish = cell.universe

        # get delta
        delta = (dish.get_upper_bounds() - dish.get_bottom_bounds()) / dish.get_grid()

        # get cell index
        index = cell.index

        # get position
        position = dish.get_bottom_bounds() + index * delta + delta / 2

        # get back substrates
        return self.on_position(cell, position)

    def on_position(self, cell, position):
        # default distribution
        return {s for s in self._container}


# ===========
#
# bio chemical reactor
#
# ===========


class BioReactor(Reactor):
    def __init__(self, **kwargs):
        # initialize super class
        super(BioReactor, self).__init__(**kwargs)

    @staticmethod
    def get_c_bonds(substrate):
        elements, count = list(substrate), 0

        # count bounds
        for i in range(0, len(elements) - 1):
            if elements[i] == "C" and elements[i + 1] == "C":
                count += 1

        # total C bonds
        return count

    @staticmethod
    def get_cn_bonds(substrate):
        return substrate.count("CN")

    @staticmethod
    def get_nc_bonds(substrate):
        return substrate.count("NC")

    @staticmethod
    def is_c_molecule(substrate):
        return all([c == "C" for c in substrate])

    @staticmethod
    def is_cn_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("CN") == len(substrate) // 2

    @staticmethod
    def is_nc_molecule(substrate):
        return len(substrate) % 2 == 0 and substrate.count("NC") == len(substrate) // 2

    def is_reactive(self, medium, a, b):
        if a == "R" or b == "R":
            return False

        if len(a + b) - 1 <= self._max_bonds:
            return True

        if self.is_cn_molecule(a) and self.is_cn_molecule(b) and self.get_cn_bonds(a + b) <= self._max_bonds:
            return True

        if self.is_nc_molecule(a) and self.is_nc_molecule(b) and self.get_nc_bonds(a + b) <= self._max_bonds:
            return True

        return False

    def can_break(self, a):
        return len(a) > 1


# ===========
#
# bio chemicals
#
# ===========


class BioChemical(Chemical):
    def __init__(self, warps, max_bonds=MAX_BONDS, diffusion=0.05, **kwargs):
        # super chemical
        super(BioChemical, self).__init__(**kwargs)

        # set initial condition
        self.setup(Substrate(name="C", diffusion=diffusion), warps)
        self.setup(Substrate(name="N", diffusion=diffusion), warps)
        self.setup(Substrate(name="R", diffusion=5 * diffusion), 0)
        self.setup(Substrate(name="Z", diffusion=diffusion), 0)

        # max bonds
        self._max_bonds = max_bonds

        # set energy token
        self.set_token("Z")

    def get_random_molecule(self):
        # random size
        size = random_state.randint(3, 2 * self._max_bonds)

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

    def get_random_reaction(self):
        # get random molecule
        random_molecule = self.get_random_molecule()

        # get breakage point
        point = random_state.randint(1, len(random_molecule))

        # return reaction tuple
        return random_molecule[:point], random_molecule[point:]

    def get_reactor(self, **kwargs):
        # craft reactor
        return BioReactor(max_bonds=self._max_bonds, **kwargs)

    @staticmethod
    def get_reaction_energy(substrate):
        return len(substrate) - 1

    @staticmethod
    def get_bond_energy(substrate):
        elements, count = list(substrate), 0

        # count bounds
        for i in range(0, len(elements) - 1):
            if elements[i] == "C" and elements[i + 1] == "C":
                count += 1

        # total C bonds
        return count

    def on_top_position(self, cell, position):
        # get universe
        universe = cell.universe

        # upper and bottom bounds
        upper_bound, lower_bound = universe.get_upper_bounds(), universe.get_bottom_bounds()

        # get delta
        delta = (upper_bound - lower_bound) / universe.get_grid()

        # distribute warps in the upper part of the universe
        if upper_bound[1] - delta[1] < position[1] < upper_bound[1]:
            return {s for s in self._container}

        # default
        return set()

# ===========
#
# antibiotics reactor
#
# ===========


class AntiBioReactor(BioReactor):
    # antibiotic molecule
    ANTIBIOTIC_RIGHT = "CNNC"
    ANTIBIOTIC_LEFT = "NCCN"

    # warps dissipation
    DISSIPATED_MOLES = 15

    def __init__(self, **kwargs):
        # antibiotics molecule
        self._antibiotic_right = AntiBioReactor.ANTIBIOTIC_RIGHT
        self._antibiotic_left = AntiBioReactor.ANTIBIOTIC_LEFT

        # initialize super class
        super().__init__(**kwargs)

    def polymerization(self, medium, a, b, amount):
        if a == self._antibiotic_right or b == self._antibiotic_right:
            # both are antibiotics
            if a == b:
                return dict()

            # break molecule
            if a == self._antibiotic_right and self.is_cn_molecule(b):
                return self.breakage(medium, b, amount)
            elif b == self._antibiotic_right and self.is_cn_molecule(a):
                return self.breakage(medium, a, amount)
            else:
                return dict()

        if a == self._antibiotic_left or b == self._antibiotic_left:
            # both are antibiotics
            if a == b:
                return dict()

            # break molecule
            if a == self._antibiotic_left and self.is_nc_molecule(b):
                return self.breakage(medium, b, amount)
            elif b == self._antibiotic_left and self.is_nc_molecule(a):
                return self.breakage(medium, a, amount)
            else:
                return dict()

        return super(AntiBioReactor, self).polymerization(medium, a, b, amount)

    def breakage(self, medium, substrate, amount):
        if substrate == self._antibiotic_right:
            # consume energy
            minimal = min(medium.get_substrate("Z"), AntiBioReactor.DISSIPATED_MOLES)

            # dissipate warps
            medium.put_substrate("Z", -minimal)
            self._universe.put_substrate("Z", minimal)

            # consume energy
            minimal = min(medium.get_substrate("CNCNCNCN"), AntiBioReactor.DISSIPATED_MOLES)

            # dissipate warps
            self.breakage(medium, "CNCNCNCN", minimal)

            # no reaction
            return dict()

        if substrate == self._antibiotic_left:
            # consume energy
            minimal = min(medium.get_substrate("Z"), AntiBioReactor.DISSIPATED_MOLES)

            # dissipate warps
            medium.put_substrate("Z", -minimal)
            self._universe.put_substrate("Z", minimal)

            # consume energy
            minimal = min(medium.get_substrate("NCNCNCNC"), AntiBioReactor.DISSIPATED_MOLES)

            # dissipate warps
            self.breakage(medium, "NCNCNCNC", minimal)

            # no reaction
            return dict()

        return super(AntiBioReactor, self).breakage(medium, substrate, amount)


class AntiBioChemical(BioChemical):
    def __init__(self, warps, max_bonds=MAX_BONDS, **kwargs):
        # super chemical
        super(AntiBioChemical, self).__init__(warps, max_bonds=max_bonds, **kwargs)

    def can_diffuse_inward(self, name, medium):
        # diffuse to right coconatas
        if name == AntiBioReactor.ANTIBIOTIC_RIGHT and medium.get_substrate("CNCNCNCN") > 0:
            return True

        # diffuse to left coconatas
        if name == AntiBioReactor.ANTIBIOTIC_LEFT and medium.get_substrate("NCNCNCNC") > 0:
            return True

        # super diffusion
        return super(AntiBioChemical, self).can_diffuse_inward(name, medium)

    def get_reactor(self, **kwargs):
        # craft reactor
        return AntiBioReactor(max_bonds=self._max_bonds, **kwargs)

    def get_membrane_diffusion(self, name: str):
        if name is FilmReactor.ANTIBIOTIC_RIGHT or name is FilmReactor.ANTIBIOTIC_LEFT:
            return 0.10

        # same than super diffusion
        return super().get_membrane_diffusion(name)

# ===========
#
# film reactor
#
# ===========


class FilmReactor(AntiBioReactor):
    # antibiotic molecule
    MATRIX_MOLECULE = "E"

    # warps dissipation
    MATRIX_DIFFUSION = 0.001

    def __init__(self, **kwargs):
        # initialize super class
        super().__init__(**kwargs)

    def can_break(self, a):
        if a is FilmReactor.MATRIX_MOLECULE:
            return True

        return super().can_break(a)

    def is_reactive(self, medium, a, b):
        # hook matrix
        if a == "E" or b == "E":
            return False

        # super reactive
        return super().is_reactive(medium, a, b)


class FilmChemical(AntiBioChemical):
    DIFFUSION_FACTOR = 5

    def __init__(self, warps, max_bonds=MAX_BONDS, **kwargs):
        # super chemical
        super(FilmChemical, self).__init__(warps, max_bonds=max_bonds, **kwargs)

    def get_reactor(self, **kwargs):
        # craft reactor
        return FilmReactor(max_bonds=self._max_bonds, **kwargs)

    def get_diffusion(self, name: str, medium=None):
        # hook matrix element
        if name == FilmReactor.MATRIX_MOLECULE:
            return FilmReactor.MATRIX_DIFFUSION

        # matrix flow
        transition = MATRIX_MOLES / MATRIX_MOTILITY
        if medium is not None and medium.get_substrate(FilmReactor.MATRIX_MOLECULE) > transition:
            return (transition * super().get_diffusion(name)) / medium.get_substrate(FilmReactor.MATRIX_MOLECULE)

        # normal diffusion
        return super().get_diffusion(name)

    def can_diffuse_inward(self, name, medium):
        if name == FilmReactor.MATRIX_MOLECULE:
            return False

        return super(FilmChemical, self).can_diffuse_inward(name, medium)

    def can_diffuse_outward(self, name, medium):
        if name == FilmReactor.MATRIX_MOLECULE:
            return False

        return super(FilmChemical, self).can_diffuse_outward(name, medium)

