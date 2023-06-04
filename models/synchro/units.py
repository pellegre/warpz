from warpz.space.box import *
from enum import IntEnum
from bisect import bisect

import numpy


# random state
random_state = numpy.random.RandomState()

# nu (angle) bins
NU_BINS = 45

# automata state
STATES_SIZE = 20


class Parcel(Cell):
    def __init__(self, **kwargs):
        # initial stack of energy
        self.light, self.initial = 0, 0
        super().__init__(**kwargs)

    def step(self):
        self.light += 14.41

    def __call__(self, signals):
        self.light /= 1.0220


# ===========
#
# Screen (with a parcel mesh)
#
# ===========

class Screen(Box):
    def __init__(self, grid=10, **kwargs):
        # initialize box
        super().__init__(grid=grid, cell=Parcel, **kwargs)

        # food and energy release
        self.mesh = numpy.zeros(self._dimensions * (self._grid,), float)

    def get_mesh(self):
        # mesh
        for each in self.get_children(condition=lambda c: isinstance(c, Parcel)):
            self.mesh[each.index[1], each.index[0]] = each.light

        # get light mesh
        return self.mesh


# ===========
#
# Pointer (mobile agent controller by a group of units)
#
# ===========

class Pointer(Particle):
    def __init__(self, target=None, radius=5.0, chromo=0.7, motility=0.01, **kwargs):
        # continuous following
        self.touch = 0

        # radius and inertia
        self.radius = radius
        self.chromo = chromo

        # target
        self.target = target
        self.motility = motility

        # agent instantiation
        super().__init__(**kwargs)

    def reached_target(self):
        self.touch += 1
        return self.touch

    def is_reachable(self, target):
        # distance to target
        return numpy.linalg.norm(target.get_position() - self.get_position()) < 0.01

    def interact(self, signals):
        # optionally interact with the environment
        pass

    def __call__(self, signals: InteractionQueue):
        # movement from players
        movement = [numpy.array([math.cos(theta), math.sin(theta)])
                    for theta in signals if isinstance(theta, float)]

        if len(movement):
            # get direction from players
            direction = sum(movement)
            norm = numpy.linalg.norm(direction)

            # transport pointer
            if norm > EPSILON:
                # transport fixed distance
                yield Transport(distance=self.motility, direction=direction / norm)

            # look for target
            if self.target:
                yield Signal(action=lambda agent: self.reached_target() and agent.reached_target(),
                             target=lambda agent: self.is_reachable(agent) and agent is self.target)

            # step on cell
            yield Signal(action=lambda a: a.step(), target=self.get_cell())

        yield self.interact(signals)

# ===========
#
# Control unit (send signals to a pointer)
#
# ===========


class ControlUnit(Agent):
    @staticmethod
    def get_angle(direction):
        # get coordinates from direction
        x, y = direction[0], direction[1]

        if y >= 0:
            # first and second quadrant
            return math.acos(x)
        else:
            return 2 * math.pi - math.acos(x)

    @staticmethod
    def get_circular_distance(one, other):
        return min(math.fabs(one - other), 2 * math.pi - math.fabs(one - other))

    def __init__(self, pointer, escape=False, target=None, direction=random_state.uniform(0, 2 * math.pi), **kwargs):
        # pointer
        self.pointer = pointer
        self.previous_position = self.pointer.get_position()

        # target
        self.target = target

        # action taken
        self.action_taken = None

        # player direction (theta angle from x-axis)
        self.theta = direction

        # if the unit should escape from the target or follow the target
        self.escape = escape

        # store previous (initial) state
        self.previous_state = self.get_state()

        # epsilon (switching from greedy to random behavior)
        self.epsilon = 0.2

        # agent instantiation
        super().__init__(**kwargs)

    def get_state(self):
        # direction
        raise RuntimeError("get_state not implemented")

    def take_action(self, state):
        # direction
        raise RuntimeError("take_action not implemented")

    def update_state(self):
        # direction
        raise RuntimeError("take_action not implemented")

    def press(self):
        # send press signal to the pointer
        yield Signal(action=self.theta, target=self.pointer)

    def action(self):
        # sample and press button
        if random_state.uniform(0.0, 1.0) < self.epsilon:
            # random move
            if random_state.uniform(0.0, 1.0) >= 0.50:
                # press button
                self.action_taken = Action.MOVE
                yield self.press()
            else:
                # idle
                self.action_taken = Action.IDLE

            # store previous state
            self.previous_state = self.get_state()
        else:
            # target direction
            state = self.get_state()

            # choose action by maximizing utility
            self.action_taken = self.take_action(state)

            # press button
            if self.action_taken is Action.MOVE:
                yield self.press()

            # store previous internal state
            self.previous_state = state

    def sense(self):
        # update internal state based on the new position
        self.update_state()

        # track position
        self.previous_position = self.pointer.get_position()

    def __call__(self, signals):
        # sense pointer position and update belief
        self.sense()

        # take action, based on belief
        yield self.action()


# ===========
#
# bayesian player
#
# ===========

class Action(IntEnum):
    MOVE = 0
    IDLE = 1


class BeliefDistribution:
    def __init__(self, bins=30):
        # belief distribution
        self.nu = numpy.linspace(0, 2.0 * math.pi, bins + 1)

        # uniform distribution
        self.count = numpy.array([1 for _ in range(0, bins)])
        self.bins = len(self.count)

        # density
        self.density = None
        self.balance()

        # initial entropy (maximum since is uniform)
        self.initial_entropy = self.entropy()
        self.minimum_entropy = -1.0 * math.log(self.bins / (2 * math.pi))

    def balance(self):
        self.density = numpy.array([self.count[i] / (2.0 * math.pi) for i in range(0, self.bins)])
        self.density *= self.bins / sum(self.count)

    def probability(self, nu):
        index = bisect(self.nu, nu) - 1
        return max(self.density[index], EPSILON)

    def integral(self, function=lambda nu: 1.0):
        accumulated = 0

        # function integral over the probability density function
        for i in range(0, len(self.nu) - 1):
            nu = (self.nu[i + 1] + self.nu[i]) / 2.0
            accumulated += function(nu) * self.density[i] * (self.nu[i + 1] - self.nu[i])

        return accumulated

    def entropy(self):
        return self.integral(function=lambda nu: -1.0 * math.log(self.probability(nu)))

    def observation(self, theta):
        # get current
        index = min(bisect(self.nu, theta) - 1, len(self.count) - 1)
        self.count[index] += 1

        # re-balance
        self.balance()

    def discard(self, theta):
        # get current
        index = min(bisect(self.nu, theta) - 1, len(self.count) - 1)
        if self.count[index]:
            # re-balance
            self.count[index] -= 1
            self.balance()


class BayesUnit(ControlUnit):
    def __init__(self, **kwargs):
        # theta belief distribution
        self.belief = BeliefDistribution(bins=NU_BINS)

        # control unit instantiation
        super().__init__(**kwargs)

    def get_state(self):
        # (the state is the angle between the agent and the target)
        direction = self.target.get_position() - self.pointer.get_position()
        return self.get_angle(direction / numpy.linalg.norm(direction))

    def take_action(self, phi):
        # choose best action maximizing the utility
        return max({Action.MOVE, Action.IDLE}, key=lambda action: self.belief.integral(
                function=lambda nu: self.utility(action, phi, nu)))

    def utility(self, action, phi, nu):
        if action is Action.MOVE:
            if not self.escape:
                return (math.pi / 2.0) - self.get_circular_distance(phi, nu)
            else:
                return self.get_circular_distance(phi, nu) - (math.pi / 2.0)
        return 0

    def update_state(self):
        # get direction of movement
        direction = self.pointer.get_position() - self.previous_position
        norm = numpy.linalg.norm(direction)

        # detect movement
        if norm > 0:
            # normalize direction
            direction /= norm
            theta = self.get_angle(direction)

            # collect evidence
            if self.action_taken is Action.MOVE:
                # accumulate evidence towards a given angle
                self.belief.observation(theta)
            else:
                self.belief.discard(theta)


# ===========
#
# automata control unit
#
# ===========


class AutomataUnit(ControlUnit):
    def __init__(self, pointer, **kwargs):
        # state distribution
        self.phi_state = numpy.linspace(0, 2.0 * math.pi, STATES_SIZE + 1)

        # boundaries states
        self.lower, self.upper = 0, pointer.get_universe().get_grid() - 1

        # Q learning matrix
        self.q_matrix = numpy.zeros((STATES_SIZE, 2))

        # Q parameters
        self.gamma = 0.60
        self.alpha = 0.95

        # agent instantiation
        super().__init__(pointer=pointer, **kwargs)

    def in_boundary(self):
        return self.pointer.get_cell().index[0] in {self.lower, self.upper} or \
               self.pointer.get_cell().index[1] in {self.lower, self.upper}

    def get_state(self):
        # direction
        direction = self.target.get_position() - self.pointer.get_position()
        phi = self.get_angle(direction / numpy.linalg.norm(direction))

        # phi (discrete) state
        phi_state = min(bisect(self.phi_state, phi), STATES_SIZE - 1)

        return phi_state

    def take_action(self, state):
        # choose best action in the current state
        return max({Action.MOVE, Action.IDLE}, key=lambda action: self.q_matrix[state, int(action)])

    def update_state(self):
        if self.action_taken is not None:
            # get direction of movement

            delta = numpy.linalg.norm(self.previous_position - self.target.get_position()) - \
                    numpy.linalg.norm(self.pointer.get_position() - self.target.get_position())

            # reward depending on the type of controller
            reward = delta if delta > 0 else -self.pointer.motility
            reward = reward if not self.escape else -reward

            # update Q matrix
            state, a = self.get_state(), int(self.action_taken)
            self.q_matrix[self.previous_state, a] = \
                self.q_matrix[self.previous_state, a] + self.alpha * \
                (reward + self.gamma * numpy.max(self.q_matrix[state, :]) - self.q_matrix[self.previous_state, a])

# ===========
#
# cell-like control unit
#
# ===========


class CellulataUnit(ControlUnit):
    def __init__(self, pointer, **kwargs):
        # state distribution
        self.phi_state = numpy.linspace(0, 2.0 * math.pi, STATES_SIZE + 1)

        # boundaries states
        self.lower, self.upper = 0, pointer.get_universe().get_grid() - 1

        # Q learning matrix
        self.q_matrix = numpy.zeros((STATES_SIZE, 2))

        # Q parameters
        self.gamma = 0.30
        self.alpha = 0.95

        # agent instantiation
        super().__init__(pointer=pointer, **kwargs)

    def in_boundary(self):
        return self.pointer.get_cell().index[0] in {self.lower, self.upper} or \
               self.pointer.get_cell().index[1] in {self.lower, self.upper}

    def get_state(self):
        norm = numpy.linalg.norm(self.pointer.chemical_queue)
        queue_direction = self.pointer.chemical_queue / norm

        if norm > 0:
            # direction
            direction = queue_direction / norm
            phi = self.get_angle(direction / numpy.linalg.norm(direction))

            # phi (discrete) state
            phi_state = min(bisect(self.phi_state, phi), STATES_SIZE - 1)

            # calculate state
            return phi_state

        else:
            return numpy.random.randint(0, STATES_SIZE)

    def take_action(self, state):
        # choose best action in the current state
        return max({Action.MOVE, Action.IDLE}, key=lambda action: self.q_matrix[state, int(action)])

    def update_state(self):
        if self.action_taken is not None:
            # reward depending on the type of controller
            norm = numpy.linalg.norm(self.pointer.chemical_queue)
            if norm > 0:
                delta = numpy.linalg.norm(self.previous_position - self.pointer.chemical_queue) - \
                        numpy.linalg.norm(self.pointer.get_position() - self.pointer.chemical_queue)

                # reward depending on the type of controller
                reward = delta if delta > 0 else -self.pointer.motility
            else:
                reward = 0

            # update Q matrix
            state, a = self.get_state(), int(self.action_taken)
            self.q_matrix[self.previous_state, a] = \
                self.q_matrix[self.previous_state, a] + self.alpha * \
                (reward + self.gamma * numpy.max(self.q_matrix[state, :]) - self.q_matrix[self.previous_state, a])


# ===========
#
# cell-like control unit and chemical signal
#
# ===========

# chemical queue
class Chemical(Signal):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, other):
        # get source
        who = self.get_source()
        who.taken = True

        # accumulate queue
        other.chemical_queue = who.queue_position


class ChemicalUnit(Particle):
    def __init__(self, position, duration, **kwargs):
        # bug time
        self.running_time, self.duration = 0, duration
        self.queue_position = position

        # chromo
        self.chromo, self.radius = 0.20, 4  # orange

        # inertial counter
        self.movement_counter, self.inertia = 0, 100

        # random direction
        self.direction = self.get_random_direction()

        # consumed unit
        self.taken = False

        # init particle
        super().__init__(position=position, **kwargs)

    def set_direction(self, direction):
        # inertia
        if self.movement_counter >= self.inertia:
            # set new direction
            self.direction = direction

            # reset inertial counter
            self.movement_counter = 0

    @staticmethod
    def get_random_direction():
        # random direction
        theta = random_state.uniform(0, 2 * math.pi)
        return numpy.array([math.cos(theta), math.sin(theta)])

    def transport(self):
        # move
        yield Transport(0.005, self.direction)

        # new time step
        self.movement_counter += 1

    def move(self):
        # chalone running time
        self.running_time += 1

        # attempt to move, after inertial property
        if self.movement_counter >= self.inertia:
            # random direction
            self.direction = self.get_random_direction()
            self.movement_counter = 0

        # bug transport
        yield self.transport()

    def done(self):
        return self.taken or self.running_time >= self.duration

    @staticmethod
    def gone():
        # bug is done
        yield Gone()

    def is_reachable(self, other):
        distance = numpy.linalg.norm(self.get_position() - other.get_position())
        return distance <= 0.01

    def queue(self):
        # bugs mating
        yield Chemical(target=lambda other: isinstance(other, CellulaFollower) and self.is_reachable(other), capacity=1)

    def interact(self, signals):
        if not self.done():
            # move my friend
            yield self.move()

            # queue cells
            yield self.queue()

        else:
            # bug is gone
            yield self.gone()

# ===========
#
# cellula
#
# ===========


class CellulaLeader(Pointer):
    def __init__(self, **kwargs):
        # chemical queue
        self.chemical_count = 0
        self.chemical_queue = numpy.array([1.0, 1.0])

        # agent instantiation
        super().__init__(**kwargs)

    def interact(self, signals):
        # optionally interact with the environment
        norm = numpy.linalg.norm(self.chemical_queue)
        if norm > 0 and self.get_time() % 10 == 0:
            yield Replicate(born=ChemicalUnit(position=self.get_position(), duration=100))


class CellulaFollower(Pointer):
    def __init__(self, **kwargs):
        # chemical queue
        self.chemical_queue = numpy.array([0.0, 0.0])
        self.chemical_count = 0

        # agent instantiation
        super().__init__(**kwargs)

    def interact(self, signals):
        pass

