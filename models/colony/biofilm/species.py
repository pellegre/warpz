from models.colony.coconata import *


class ConjugatePump(Signal):
    def __init__(self, pump, **kwargs):
        # initialize super
        super().__init__(**kwargs)

        # store pump
        self._pump = pump

    def __call__(self, coconata):
        # get cell
        cell = coconata.get_cell()

        # replicate pump
        replicated = self._pump.replicate()

        # replicate gene
        if replicated.get_moles() > 0:
            # initialize
            replicated.init(coconata)

            # insert pump
            if coconata.get_chromosome().insert_pump(self._pump):
                # count flux
                cell.gene_flux()


class ConjugateReaction(Signal):
    def __init__(self, reaction, **kwargs):
        # initialize super
        super().__init__(**kwargs)

        # store pump
        self._reaction = reaction

    def __call__(self, coconata):
        # get cell
        cell = coconata.get_cell()

        # replicate reaction
        replicated = self._reaction.replicate()

        # replicate gene
        if replicated.get_moles() > 0:
            # initialize
            replicated.init(coconata)

            # insert pump
            coconata.get_chromosome().insert_reaction(self._reaction)

            # count flux
            cell.gene_flux()


class Specie(Coconata):
    def __init__(self, mother, container, universe, **kwargs):
        # super init
        super(Specie, self).__init__(dish=universe, chromosome=mother, container=container, **kwargs)

        # random direction (and inertia)
        self._direction = self.get_random_direction()
        self._movement_counter, self._inertia = 0, random_state.randint(10, 25)

        # moved distance
        self._transport_energy = 15

        # done
        self._done = False

        # screening
        self.radius, self.color = self.get_radius(), self.get_color()

    def get_color_by_film(self):
        # color
        return 0.05 if self.is_planktonic() else 0.60

    def get_color_by_specie(self):
        # color
        return 0.15 if (self._chromosome.get_anabolic_drive() == "CNCNCNCN") else 0.85

    def get_color_by_state(self):
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

    def get_color(self):
        return self.get_color_by_film()

    @staticmethod
    def get_random_direction():
        # random direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def get_motility_in_matrix(self):
        # matrix molecule
        matrix = self.get_cell().get_substrate(FilmReactor.MATRIX_MOLECULE)

        # cross section
        transition = MATRIX_MOLES / MATRIX_MOTILITY
        cross_section = (1 + (matrix / transition)) if (0 <= matrix < transition) \
            else (transition / matrix)

        # mean free path
        mean = self.get_universe().get_motility() * cross_section

        # coconata motility
        motility = max(0, numpy.random.normal(mean, 0.02 * mean))

        # get back motility
        return motility

    # ======
    # transport
    #
    def transport(self, units=1):
        # get motility
        motility = self.get_motility_in_matrix()

        # tokens
        if self.is_planktonic():
            token, price = self.get_universe().get_chemistry().get_token(), \
                           self.get_universe().get_chemistry().get_token_price()

            # transport tokens
            tokens = int(self._transport_energy / price)

            # take out minimal
            minimal = min(self.get_substrate(token), tokens)

            # take token
            self.put_substrate(token, -minimal)

            # balance token
            self.get_universe().put_substrate(token, minimal)

        if motility > 0:
            if not self.is_planktonic() and random_state.uniform(0, 1) < 0.20:
                # matrix push
                center = self.get_cell().get_position() - self.get_position()
                yield Transport(units * motility, Direction(vector=center).vector)
            else:
                # move
                yield Transport(units * motility, self._direction)

            # new time step
            self._movement_counter += 1

    # ======
    # move
    #
    def move(self):
        # attempt to move, after inertial property
        if self._movement_counter >= self._inertia:
            # random direction
            self._direction = self.get_random_direction()

            # reset counter
            self._movement_counter = 0

        # bug transport
        yield self.transport()

    def get_radius(self):
        return 0.005 * (self.get_moles() / CHAMBER_MOLES)

    # ======
    # done
    #
    def done(self):
        # update screening parameters
        self.radius = self.get_radius()

        # update state
        self.state()

        # setup color
        self.color = self.get_color()

        # get anabolic drive moles
        moles = self.get_substrate(self._chromosome.get_anabolic_drive())

        # check lysis and moles
        if moles == 0 or self.get_moles() > LYSIS_MOLES:
            # put on cell
            substrates = self.get_container()
            for s in substrates:
                self.get_cell().put_substrate(s, substrates[s])
                self.put_substrate(s, -substrates[s])

            # gone
            self._done = True

        # get back state
        return self._done

    # ======
    # propagate
    #
    def propagate(self):
        # replicate pumps
        for pump in self.get_chromosome().get_pumps():
            # gene replication rate
            if random_state.uniform(0, 1) < pump.get_replication_rate():
                # mutate pump
                if self.is_critical():
                    replicated = pump.replicate(random=True)
                else:
                    replicated = pump.replicate()

                if replicated.get_moles() > 0:
                    # initialize
                    replicated.init(self)

                    # insert pump gene
                    self.get_chromosome().insert_pump(replicated)

                    # conjugate it
                    self.get_cell().add_signal(ConjugatePump(pump=pump, capacity=1))

        # replicate reaction
        for reaction in self.get_chromosome().get_reactions():
            # gene replication rate
            if random_state.uniform(0, 1) < reaction.get_replication_rate() or self.is_critical():
                replicated = reaction.replicate()
                if replicated.get_moles() > 0:
                    # initialize
                    replicated.init(self)

                    # insert reaction
                    self.get_chromosome().insert_reaction(replicated)

                    # conjugate it
                    self.get_cell().add_signal(ConjugateReaction(reaction=reaction, capacity=1))

        # get a signal
        signal = self.get_cell().get_signal()

        # execute it
        if signal is not None:
            signal(self)

    # ======
    # replicate
    #
    def replicate(self):
        # get anabolic drive moles
        moles = self.get_substrate(self._chromosome.get_anabolic_drive())

        if moles > CHAMBER_MOLES // 2:
            # get substrates
            substrates = self.get_container()

            # craft container for new born
            born_container = dict()
            for s in substrates:
                # splash it
                born_container[s] = substrates[s] // 2

                # balance
                self.put_substrate(s, -born_container[s])

            # copy chromosome
            chromosome = copy.deepcopy(self._chromosome)
            chromosome.mutate()

            # replicate
            yield Replicate(born=Specie(position=self.get_position(), container=born_container,
                                        universe=self.get_universe(), mother=chromosome))

            # treat as just born
            self._born = True

    # ======
    # interact
    #
    def interact(self, signals):
        if not self.done():
            # diffuse
            self.diffuse_inward()

            # sense the environment
            self.sense()

            # catalytic reaction
            self.react_and_pump()

            # diffuse
            self.diffuse_outward()

            # move
            yield self.move()

            # coco splash
            yield self.replicate()

        else:
            # gone
            yield Gone()


class RightCoco(Specie):
    def __init__(self, universe, matrix=True, **kwargs):
        # setup gene
        chromosome = RightSpecie(matrix=matrix)

        # initial container
        container = {"CNCNCNCN": CHAMBER_MOLES // 4, "CNCNCN": CHAMBER_MOLES // 4, "CNCN": CHAMBER_MOLES // 4,
                     "CN": CHAMBER_MOLES // 4, "CCC": CHAMBER_MOLES // 16, "CCCC": CHAMBER_MOLES // 16,
                     "Z": CHAMBER_MOLES // 8}

        # super init
        super(RightCoco, self).__init__(universe=universe, mother=chromosome, container=container, **kwargs)


class LeftCoco(Specie):
    def __init__(self, universe, matrix=True, **kwargs):
        # setup gene
        chromosome = LeftSpecie(matrix=matrix)

        # initial container
        container = {"NCNCNCNC": CHAMBER_MOLES // 4, "NCNCNC": CHAMBER_MOLES // 4, "NCNC": CHAMBER_MOLES // 4,
                     "NC": CHAMBER_MOLES // 4, "CCC": CHAMBER_MOLES // 16, "CCCC": CHAMBER_MOLES // 16,
                     "Z": CHAMBER_MOLES // 8}

        # super init
        super(LeftCoco, self).__init__(universe=universe, mother=chromosome, container=container, **kwargs)
