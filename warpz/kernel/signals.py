from collections.abc import Iterable
from collections.abc import MutableMapping

import types
import math
import time

from warpz.kernel.agent import *


class Signal(object):
    """"
    signal object
    """

    SOURCE = "_source"
    TARGET = "_target"
    ENVIRONMENT = "_environment"

    VERBOSE = False

    # broadcast signaling
    class BROADCAST:
        def __init__(self, target=lambda a: True):
            self.target = target

        def __call__(self, agent):
            return self.target(agent)

    def __init__(self, action=None, target=None, source=None, capacity=math.inf, priority=1):
        """ signal instantiation

        parameters
        ----------
        action : object
            signal object
        target : destination agent ID (or callable)
            destination agent
        """

        # signal action
        self._action = action

        # destination
        if isinstance(target, Iterable):
            self._target = [self._resolve_agent(obj) for obj in target]
        else:
            self._target = self._resolve_agent(target)

        # signal's name
        self._name = self.__class__.__name__ + "[" + hex(id(self)) + "]"

        # signal's source
        self._source = source

        # signal's capacity
        self._capacity = capacity

        # signal's priority
        self._priority = priority

        # signal's environment
        self._environment = None

    def __str__(self):
        return "< signal ( " + self._name + ") ----> " + str(self._target) + " >"

    def get_environment(self):
        """ get underlying environment

        """

        return self._environment

    def get_target(self):
        """ agent signal destination (or callable)

        returns
        -------
        id : Agent, Iterable, callable
            destination agent
        """

        return self._target

    def get_priority(self):
        """ signal priority

        returns
        -------
        int : priority
            signal priority
        """

        return self._priority

    def get_source(self):
        """ agent signal source

        returns
        -------
        id : Agent
            source agent
        """
        return self._source

    def get_capacity(self):
        """ signal's capacity

        returns
        -------
        capacity : int
            signal capacity
        """

        return self._capacity

    def collect(self):
        """ collect signal's capacity

        """

        # capacity update
        self._capacity -= 1

    @staticmethod
    def _resolve_agent(agent):
        if hasattr(agent, Agent.GET_ID):
            return agent.get_id()

        return agent

    def __call__(self, *args, **kwargs):
        # interact
        if self._capacity > 0:
            if callable(self._action) and not isinstance(self._action, Agent):
                action = self._action(*args, **kwargs)
                if isinstance(action, types.GeneratorType):
                    yield from action
                elif isinstance(action, Iterable):
                    yield from action
                else:
                    # propagate stimulus after interaction
                    yield action
            else:
                # propagate stimulus
                yield self._action


# signal stack
class SignalQueue(MutableMapping):
    def __init__(self, *args, **kwargs):
        self._signal_stack = dict()
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        if isinstance(key, Iterable):
            return {agent: self._signal_stack[self.__keytransform__(key)] for agent in key}
        else:
            return self._signal_stack[self.__keytransform__(key)]

    def __setitem__(self, key, value):
        if isinstance(key, Iterable):
            for k in key:
                self.__setitem__(k, value)
        else:
            if not isinstance(value, Iterable):
                self._signal_stack[self.__keytransform__(key)] = [value]
            else:
                self._signal_stack[self.__keytransform__(key)] = value

    def __iter__(self):
        for a in self._signal_stack:
            yield from self._signal_stack[a]

    def __len__(self):
        return len(self._signal_stack)

    def __delitem__(self, key):
        if isinstance(key, Iterable):
            for agent in key:
                del self._signal_stack[self.__keytransform__(agent)]
        else:
            del self._signal_stack[self.__keytransform__(key)]

    def __contains__(self, key):
        return self.__keytransform__(key) in self._signal_stack

    @staticmethod
    def __keytransform__(key):
        if hasattr(key, Agent.GET_ID):
            return key.get_id()
        else:
            return key

    def append(self, value, priority=1):
        if not isinstance(value, Iterable) and isinstance(value, Signal):
            key = priority
            if self.__keytransform__(key) not in self._signal_stack:
                self._signal_stack[self.__keytransform__(key)] = [value]
            else:
                self._signal_stack[self.__keytransform__(key)].append(value)
        else:
            for each in value:
                self.append(each)

    def add(self, other):
        assert isinstance(other, SignalQueue)
        for a in other._signal_stack:
            if a in self._signal_stack:
                self._signal_stack[self.__keytransform__(a)] += other._signal_stack[a]
            else:
                self._signal_stack[self.__keytransform__(a)] = other._signal_stack[a]

    def flush(self):
        self._signal_stack.clear()


class InteractionQueue(SignalQueue):
    PRECURSOR = -1
    PRIORITY = 0

    def __init__(self, entity, *args, **kwargs):
        self.entity = entity
        self.interactions = []
        super(InteractionQueue, self).__init__(*args, **kwargs)

    def __getitem__(self, key):
        signals = self._signal_stack[self.__keytransform__(key)]

        for signal in signals:
            if signal.get_capacity() > 0 and (not signal.get_source() or (signal.get_source().exists())):

                if not callable(signal.get_target()) or \
                        (callable(signal.get_target()) and (signal.get_source() is not self.entity or
                                                            isinstance(signal.get_target(), Signal.BROADCAST)) and
                         signal.get_target()(self.entity)):

                    if not signal.get_source():
                        raise EnvironmentError("can't run sourceless signal")

                    start = time.process_time_ns()
                    output = signal(self.entity)
                    schwifties = time.process_time_ns() - start

                    if isinstance(signal.get_source(), Agent):
                        agent = signal.get_source()
                    else:
                        agent = self.entity.get_environment().get_agent(signal.get_source())

                    agent.add_schwifties(schwifties)

                    if Signal.VERBOSE:
                        print("[  ---->] running signal (__getitem__) " + type(signal).__name__ + "@" +
                              str(signal.get_source()) + " - " + str(schwifties))

                    if isinstance(output, Iterable):
                        for interaction in filter(lambda i: i is not None, output):
                            if isinstance(interaction, Signal):
                                self.interactions.append(interaction)
                            elif isinstance(interaction, Iterable):
                                for each in interaction:
                                    self.interactions.append(each)
                            else:
                                yield interaction

                    signal.collect()

        self.__delitem__(key)

    def __iter__(self):
        while len(self._signal_stack) > 0:

            a = next(iter(sorted(self._signal_stack)))

            for signal in self._signal_stack[a]:
                if signal.get_capacity() > 0 and (not signal.get_source() or (signal.get_source().exists())):

                    if not callable(signal.get_target()) or \
                            (callable(signal.get_target()) and (signal.get_source() is not self.entity or
                                                                isinstance(signal.get_target(), Signal.BROADCAST)) and
                             signal.get_target()(self.entity)):

                        if not signal.get_source():
                            raise EnvironmentError("can't run sourceless signal")

                        start = time.process_time_ns()
                        output = signal(self.entity)
                        schwifties = time.process_time_ns() - start

                        if isinstance(signal.get_source(), Agent):
                            agent = signal.get_source()
                        else:
                            agent = self.entity.get_environment().get_agent(signal.get_source())

                        agent.add_schwifties(schwifties)

                        if Signal.VERBOSE:
                            print("[  ---->] running signal (__iter__) " + type(signal).__name__ + "@" +
                                  str(signal.get_source()) + " - " + str(schwifties))

                        if isinstance(output, Iterable):
                            for interaction in filter(lambda i: i is not None, output):
                                if isinstance(interaction, Signal):
                                    self.interactions.append(interaction)
                                elif isinstance(interaction, Iterable):
                                    for each in interaction:
                                        self.interactions.append(each)
                                else:
                                    yield interaction
                        else:
                            yield output

                        signal.collect()

            self.__delitem__(a)
