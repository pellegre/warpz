import numpy
import math
import copy

from collections.abc import Iterable

# ===========
#
# gene
#
# ===========


class Gene:
    def __init__(self, number, pool):
        self._pool = pool
        self._traits = numpy.zeros(number, int)

    def __str__(self):
        string = str()
        genes = self._pool.get_genes()
        for trait in genes:
            string += self._get_gene_string(trait, genes[trait])
        return string

    def _get_gene_string(self, trait, value):
        string = str()

        if isinstance(value, Iterable):
            string += str(trait) + "->" + "{:03}".format(self._traits[trait]) + "|"

        return string

    def __hash__(self):
        return hash(frozenset({(i, e) for i, e in enumerate(self._traits)}))

    def __eq__(self, other):
        return numpy.all(self._traits == other.get_traits())

    def __ge__(self, other):
        return numpy.all(self._traits >= other.get_traits())

    def crossover(self, other, random_state):
        # new gene
        new = Gene(number=len(self._traits), pool=self._pool)

        # average crossover
        genes = self._pool.get_genes()
        for trait in genes:
            if isinstance(genes[trait], tuple):
                new._traits[trait] = ((self._traits[trait] + other.get_traits()[trait]) / 2).astype(int)
            elif isinstance(genes[trait], Iterable):
                # sample resulting allele
                if random_state.randint(0, 2):
                    new._traits[trait] = self._traits[trait]
                else:
                    new._traits[trait] = other.get_traits()[trait]
            else:
                # single valued
                new._traits[trait] = self._traits[trait]

        # cross it over
        return new

    def distance(self, other):
        return self._pool.distance(self, other)

    def get_traits(self):
        return self._traits

    def get_genes(self):
        return self._pool.get_genes()

    def mutate(self, random_state):
        # get random gene
        random_gene = self._pool.get_random_gene(random_state)

        # trait locus
        locus = random_state.randint(0, len(self._traits))
        self._traits[locus] = random_gene.get_traits()[locus]

    def set_trait(self, trait, value):
        self._traits[trait] = value

    def get_trait(self, trait):
        return self._traits[trait]

    def has_trait(self, trait):
        return self._traits[trait] > (self._pool.get_capacity() / 2)

    def get_capacity(self):
        return self._pool.get_capacity()


class GenePool:
    def __init__(self, capacity, genes):
        self._genes = genes
        self._capacity = capacity

        self._total_range = 0
        for each in self._genes:
            value = self._genes[each]
            self._total_range += self._get_range_value(value)

    @staticmethod
    def _get_range_value(value):
        total_range = 0

        if isinstance(value, tuple):
            total_range += math.fabs(max(value) - min(value))
        elif isinstance(value, Iterable):
            total_range += len(value)

        return total_range

    def get_genes(self):
        return self._genes

    def get_random_gene(self, random_state):
        # create random gene
        gene = Gene(number=len(self._genes), pool=self)

        # go over each trait
        for each in self._genes:
            # set trait value
            value = self._genes[each]
            self._set_trait_value(random_state, gene, each, value)

        return gene

    def _set_trait_value(self, random_state, gene, trait, value):
        if isinstance(value, tuple):
            # set random value in range
            gene.set_trait(trait, random_state.randint(value[0], value[1] + 1))
        elif isinstance(value, Iterable):
            # choose randomly an interval
            interval = random_state.randint(0, len(value))
            self._set_trait_value(random_state, gene, trait, value[interval])
        else:
            # setup value
            gene.set_trait(trait, value)

    def get_capacity(self):
        return self._capacity

    def get_chromo(self, gene):
        chromo = 0
        for each in self._genes:
            value = self._genes[each]
            chromo += self._get_chromo_value(gene, each, value)

        # return colorful gene
        return chromo

    def _get_chromo_value(self, gene, trait, value):
        chromo = 0

        if isinstance(value, tuple):
            chromo += (gene.get_trait(trait) - min(value)) / self._total_range
        elif isinstance(value, Iterable):
            chromo += 1 / self._total_range

        return chromo

    def distance(self, one, other):
        # accumulate distance
        dist = 0.0

        # compare each trait
        for each in self._genes:
            value = self._genes[each]
            if isinstance(value, tuple):
                dist += math.fabs(one.get_trait(each) - other.get_trait(each)) / self._total_range
            elif isinstance(value, Iterable):
                dist += min(1.0, math.fabs(one.get_trait(each) - other.get_trait(each))) / self._total_range

        # return normalized distance
        dist = dist
        assert 0 <= dist <= 1
        return dist


class GeneFrame(GenePool):
    def __init__(self, seed_pool, genes_frame, knock_genes=None):
        # genes frame
        self.genes_frame = genes_frame

        # skip genes from population
        if knock_genes is None:
            self.knock_genes = dict()
        else:
            self.knock_genes = knock_genes

        # traits mapping
        self.traits_map = {str(t): t for t in seed_pool}

        # get gene limits
        genes = {self.traits_map[g]: (self.genes_frame[g].min(), self.genes_frame[g].max())
                 if self.traits_map[g] not in self.knock_genes else self.knock_genes[self.traits_map[g]]
                 for g in self.genes_frame}

        # initialize pool
        super(GeneFrame, self).__init__(genes=genes, capacity=int(self.genes_frame.max().values[0]))

    def get_random_gene(self, random_state):
        # create random gene
        gene = Gene(number=len(self._genes), pool=self)

        # select a random gene from the population
        idx = random_state.randint(0, len(self.genes_frame))
        row = self.genes_frame.iloc[idx]

        # go over each trait
        for each in self.genes_frame.columns:
            # get trait
            trait = self.traits_map[each]

            # check knock off genes
            if trait in self.knock_genes:
                # set trait value
                value = self._genes[trait]
                self._set_trait_value(random_state, gene, trait, value)
            else:
                # set trait value
                gene.set_trait(trait, int(row[each]))

        return gene

