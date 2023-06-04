from enum import IntEnum

from models.gene.pool import *

import pickle


# ===========
#
# Gene Pool
#
# ===========

# automata state
GENE_CAPACITY = 256


# cell traits
class Trait(IntEnum):
    BORN = 0
    MOTILITY = 1
    GREEDINESS = 2
    STATES = 3
    EPSILON = 4
    ALPHA = 5
    GAMMA = 6
    UNITS = 7
    MUTATION = 8
    RELEASE = 9
    RADIUS = 10
    BIPOLARITY = 11
    CROSSOVER = 12
    CONSUMPTION = 13
    PRODUCTION = 14
    TOTAL = 15


# max motility in genes
MAX_MOTILITY = 7

# bipolarity factor
BIPOLARITY_FACTOR = 1

# seed gene pool
seed_gene_pool = {
    # born energy
    Trait.BORN: (70, GENE_CAPACITY),

    # cell motility
    Trait.MOTILITY: (4, MAX_MOTILITY),

    # cell greediness
    Trait.GREEDINESS: (0, 7),

    # control units states
    Trait.STATES: (4, 24),

    # control greediness
    Trait.EPSILON: (0, GENE_CAPACITY // 2),

    # learning rate
    Trait.ALPHA: (GENE_CAPACITY // 3, GENE_CAPACITY),

    # discount factor
    Trait.GAMMA: (GENE_CAPACITY // 3, GENE_CAPACITY),

    # total control units
    Trait.UNITS: (4, 24),

    # mutation rate
    Trait.MUTATION: (0, GENE_CAPACITY // 4),

    # chemokine release rate
    Trait.RELEASE: (0, GENE_CAPACITY),

    # consumption of chemokines
    Trait.CONSUMPTION: (0, GENE_CAPACITY),

    # production of chemokines
    Trait.PRODUCTION: (0, GENE_CAPACITY),

    # bipolarity threshold
    Trait.BIPOLARITY: (0, GENE_CAPACITY),

    # cell radius (size)
    Trait.RADIUS: (50, GENE_CAPACITY),

    # crossover probability
    Trait.CROSSOVER: (0, GENE_CAPACITY)
}

# releasing wave
releasing_wave_gene_pool = {
    # born energy
    Trait.BORN: (70, GENE_CAPACITY),

    # cell motility
    Trait.MOTILITY: (4, MAX_MOTILITY),

    # cell greediness
    Trait.GREEDINESS: (0, 3),

    # control units states
    Trait.STATES: (4, 24),

    # control greediness
    Trait.EPSILON: (0, GENE_CAPACITY // 2),

    # learning rate
    Trait.ALPHA: (GENE_CAPACITY // 3, GENE_CAPACITY),

    # discount factor
    Trait.GAMMA: (GENE_CAPACITY // 3, GENE_CAPACITY),

    # total control units
    Trait.UNITS: (4, 24),

    # mutation rate
    Trait.MUTATION: (0, 15),

    # chemokine release rate
    Trait.RELEASE: (220, GENE_CAPACITY),

    # consumption of chemokines
    Trait.CONSUMPTION: (0, 30),

    # production of chemokines
    Trait.PRODUCTION: (0, 30),

    # bipolarity threshold
    Trait.BIPOLARITY: (150, 200),

    # cell radius (size)
    Trait.RADIUS: (180, GENE_CAPACITY),

    # crossover probability
    Trait.CROSSOVER: (90, 120)
}


class GenePopulation(GeneFrame):
    def __init__(self, folder, run_id, tail=1000, **kwargs):
        # set simulation environment
        self.folder = folder
        self.run_id = run_id

        # get gene frame
        tally_frame = self.get_simulation_distribution()
        genes_frame = tally_frame.genes_frame[[c for c in tally_frame.genes_frame if c != "time"]]
        print("[+] opening gene pool from population", self.run_id)

        super(GenePopulation, self).__init__(genes_frame=genes_frame.tail(tail), seed_pool=seed_gene_pool, **kwargs)

    def get_simulation_distribution(self):
        return pickle.load(open(self.folder + "/run-" + str(self.run_id) + "/surgery.pkl", "rb"))
