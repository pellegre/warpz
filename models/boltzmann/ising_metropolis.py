from warpz.kernel.universe import *

import numpy

EPSILON = 10E-10


class Reservoir(Universe):
    def __init__(self, **kwargs):
        # initialize base signal
        super().__init__(**kwargs)

    def __call__(self, signals):
        # random flip
        for each in self.get_children():
            if isinstance(each, Matrix):
                yield Flip(target=each)


class Matrix(Agent):
    def __init__(self, size, j=0.30, **kwargs):
        # grid size
        self.l = size

        # interaction (units of kT)
        self.j = j

        # initialize base agent
        super().__init__(**kwargs)

        # spin sites
        self.material = numpy.random.randint(2, size=(self.l, self.l))
        self.material[self.material == 0] = -1

        # energy of the matrix
        self.energy = self.get_energy() - self.get_min_energy()

    def get_energy_for_site(self, i, j, spin):
        energy = -self.j * spin * (self.material[i, (j - 1) % self.l] +
                                   self.material[i, (j + 1) % self.l] +
                                   self.material[(i - 1) % self.l, j] +
                                   self.material[(i + 1) % self.l, j])

        return energy

    def get_energy(self):
        right_shift, up_shift = numpy.zeros((self.l, self.l), dtype=int), numpy.zeros((self.l, self.l), dtype=int)

        right_shift[:, 0] = self.material[:, self.l - 1]
        right_shift[:, 1:self.l] = self.material[:, 0:(self.l - 1)]

        up_shift[self.l - 1, :] = self.material[0, :]
        up_shift[0:(self.l - 1), :] = self.material[1:self.l, :]

        return -self.j * (sum(sum(numpy.multiply(right_shift, self.material))) +
                          sum(sum(numpy.multiply(up_shift, self.material))))

    def get_min_energy(self):
        return -self.j * 2 * (self.l * self.l)

    def get_energy_after_flip(self, i, j):
        return self.energy - \
               self.get_energy_for_site(i, j, self.material[i, j]) + \
               self.get_energy_for_site(i, j, (-1) * self.material[i, j])

    def get_magnetic_order(self):
        up = sum(sum((self.material * 1) == 1))
        sites = self.l * self.l

        return (2 * up - sites) / sites


class Flip(Signal):
    def __init__(self, **kwargs):
        # initialize base signal
        super().__init__(**kwargs)

    def __call__(self, matrix: Matrix):
        # choose some random site
        i, j = numpy.random.randint(0, matrix.l), numpy.random.randint(0, matrix.l)

        # get energy delta after flip
        energy_after_flip = matrix.get_energy_after_flip(i, j)
        delta_energy = energy_after_flip - matrix.energy

        # metropolis algorithm for state acceptance
        if delta_energy < 0 or numpy.random.random() < math.exp(-delta_energy):
            # toggle spin
            matrix.material[i, j] = (-1) * matrix.material[i, j]

            # update matrix energy
            matrix.energy = energy_after_flip


def main():
    print("[+] ising (with metropolis)")

    # initialize matrix
    matrix = Matrix(size=50, j=1.00)
    reservoir = Reservoir()

    # interaction constant
    print("[-] interaction = ", matrix.j)

    # add matrix
    reservoir.add(matrix)

    flips, steps = 0, 5000000
    energy, magnetic_order = 0, 0
    for k in range(1, steps):
        # run system
        reservoir.run()

        # tally observables
        energy += matrix.energy
        magnetic_order += matrix.get_magnetic_order()

        if k % 10000 == 0:
            print("[+] step <", k, ">")
            print(f"[=] magnetic order (average) : {magnetic_order / k:.2f}")
            print(f"[=] energy (average)         : {energy / k:.2f}")
            print(f"[=] state magnetic order     : {matrix.get_magnetic_order():.2f}")
            print(f"[=] state energy             : {matrix.energy / k:.2e}")
            print(f"[=] schwifty                 : {reservoir.get_schwifties() / k:.2f}")


main()
