from warpz.kernel.universe import *

import numpy

EPSILON = 10E-10


class Reservoir(Universe):
    def __init__(self, capacity=0, energy=0, **kwargs):
        # initial reservoir energy
        self.energy = energy

        # reservoir capacity (can't take more than this amount of energy)
        self.capacity = capacity

        # reservoir capacity (can't take more than this amount of energy)
        self.overflow = 0

        # current spin flips
        self.flips = 0

        # initialize base signal
        super().__init__(**kwargs)

    def __call__(self, signals):
        # random flip
        for each in self.get_children():
            if isinstance(each, Matrix):
                yield Flip(reservoir=self, target=each)


class Matrix(Agent):
    def __init__(self, size, j=1.00, random=False, **kwargs):
        # grid size
        self.l = size

        # interaction (units of kT)
        self.j = j

        # initialize base agent
        super().__init__(**kwargs)

        # spin sites
        if random:
            self.material = numpy.random.randint(2, size=(self.l, self.l))
            self.material[self.material == 0] = -1
        else:
            self.material = numpy.ones((self.l, self.l))

        # energy of the matrix
        self.interactions = self.get_interactions()

    def get_interactions_for_site(self, i, j, spin):
        energy = (-1) * spin * (self.material[i, (j - 1) % self.l] +
                                self.material[i, (j + 1) % self.l] +
                                self.material[(i - 1) % self.l, j] +
                                self.material[(i + 1) % self.l, j])

        return energy

    def get_interactions(self):
        right_shift, up_shift = numpy.zeros((self.l, self.l), dtype=int), numpy.zeros((self.l, self.l), dtype=int)

        right_shift[:, 0] = self.material[:, self.l - 1]
        right_shift[:, 1:self.l] = self.material[:, 0:(self.l - 1)]

        up_shift[self.l - 1, :] = self.material[0, :]
        up_shift[0:(self.l - 1), :] = self.material[1:self.l, :]

        return (-1) * (sum(sum(numpy.multiply(right_shift, self.material))) +
                       sum(sum(numpy.multiply(up_shift, self.material)))) - self.get_min_interactions()

    def get_min_interactions(self):
        return (-2) * (self.l * self.l)

    def get_interactions_after_flip(self, i, j):
        return self.interactions - \
               self.get_interactions_for_site(i, j, self.material[i, j]) + \
               self.get_interactions_for_site(i, j, (-1) * self.material[i, j])

    def get_energy_after_flip(self, i, j):
        return self.j * self.get_interactions_after_flip(i, j)

    def get_energy(self):
        return self.j * self.interactions

    def get_magnetic_order(self):
        up = sum(sum((self.material * 1) == 1))
        sites = self.l * self.l

        return (2 * up - sites) / sites

    def flip(self, i, j):
        self.interactions = self.get_interactions_after_flip(i, j)
        self.material[i, j] = (-1) * self.material[i, j]


class Flip(Signal):
    def __init__(self, reservoir: Reservoir, **kwargs):
        # reservoir reference
        self.reservoir = reservoir

        # initialize base signal
        super().__init__(**kwargs)

    def __call__(self, matrix: Matrix):
        # choose some random site
        ix, jx = numpy.random.randint(0, matrix.l), numpy.random.randint(0, matrix.l)

        # get energy delta after flip
        energy_after_flip = matrix.get_energy_after_flip(ix, jx)
        delta_energy = energy_after_flip - matrix.get_energy()

        # heat flow to / from the reservoir
        heat_flow = -delta_energy

        # state energy delta transition
        if delta_energy < 0:
            # heat flows to the reservoir
            if self.reservoir.energy < matrix.get_energy():
                # check overflow
                if self.reservoir.energy + heat_flow <= self.reservoir.capacity:
                    # toggle spin
                    matrix.flip(ix, jx)

                    # count flips
                    self.reservoir.flips += 1

                    # heat flow
                    self.reservoir.energy += heat_flow

                else:
                    # reservoir overflow
                    self.reservoir.overflow += 1

        elif delta_energy > 0:
            # heat should flow from the reservoir
            if self.reservoir.energy > matrix.get_energy() and self.reservoir.energy >= delta_energy:
                # toggle spin
                matrix.flip(ix, jx)

                # count flips
                self.reservoir.flips += 1

                # heat flow
                self.reservoir.energy += heat_flow

        else:
            # toggle spin
            matrix.flip(ix, jx)

            # count flips
            self.reservoir.flips += 1


def main():
    print("[+] ising (with metropolis)")

    # initialize matrix
    matrix = [Matrix(size=50, j=1), Matrix(size=50, j=1)]
    reservoir = Reservoir(capacity=math.inf, energy=1200)

    print("[-] capacity = ", reservoir.capacity)

    # add matrix
    for each in matrix:
        reservoir.add(each)

    flips, steps = 0, 5000000
    reservoir_energy = 0
    for k in range(1, steps):
        # run system
        reservoir.run()

        # tally observables
        reservoir_energy += reservoir.energy

        # print some stats
        if k % 10000 == 0 or k == 1:

            print("[+] step <", k, ">")
            print(f"[=] reservoir energy (average) : {reservoir_energy / k:.2f}")
            print("[=] spin flips                 :", reservoir.flips)

            print(f"[=] reservoir energy           : {reservoir.energy:.2f}")
            print("[=] reservoir overflow         :", reservoir.overflow)
            print(f"[=] schwifty                   : {reservoir.get_schwifties() / k:.2f}")

            matrix_energy = 0
            for i, each in enumerate(matrix):
                print("[+]   -------> matrix ", i)
                print(f"[#] state magnetic order       : {each.get_magnetic_order():.2f}")
                print(f"[#] state energy               : {each.get_energy():.2f}")

                matrix_energy += each.get_energy()

            print("[+]   -------> total ")
            print(f"[=] total energy               : {matrix_energy + reservoir.energy:.2f}\n")


main()
