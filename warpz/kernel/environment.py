import networkx as nx
import time

from collections.abc import Iterable
from warpz.kernel.core import *
from warpz.kernel.agent import Agent


class One(Agent):
    def __init__(self):
        super().__init__(leader=True)
        self._name = self.__class__.__name__


# environment, the good old friend
class Environment(object):
    """"
    Environment (agent's scheduler)
    """

    MULTICAST = "multicast"
    UNICAST = "unicast"
    AGENT = "agent"

    VERBOSE = False

    def __init__(self, one=None):
        # agent's tree (AID to data structures)
        self._tree = nx.DiGraph()

        # signal's tree (output and input flow)
        self._flow = nx.DiGraph()

        # initial time
        self._time = 0

        # init done
        self._init_done = True

        # one
        self._one = One() if not one else one
        self.add(self._one)

        # periodic flows
        self._flow_cycle = dict()
        self._flow_cycle_period = 0

        # cache
        self._cache_agent_leaders = dict()

    def get_schwifties(self):
        """ get environment's accumulated schwifties

        """

        return self._schwifties

    def add_schwifties(self, schwifties):
        """ accumulate schwifties

        """

        self._schwifties += schwifties

    def take_schwifties(self, schwifties):
        """ subtract schwifties

        """

        self._schwifties -= schwifties

    def get_time(self):
        """ agent's current time

        returns
        -------
        time : int
            agent's time
        """

        return self._time

    def get_count(self):
        """ agent count

        returns
        -------
        count : int
            amount of agents
        """

        return len(self._tree.nodes)

    def get_flow(self):
        """
        get flow from executor

        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("flow is not available before calling Environment.__init__()")

        return self._flow

    def add(self, agent):
        """
        add a new agent to this environment

        parameters
        ----------
        agent : Agent
            agent_instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("adding agents is not supported before calling Environment.__init__()")

        if isinstance(agent, Agent):
            # sync it with the one
            setattr(agent, Agent.ENVIRONMENT, self)

            # add it to the tree
            self._tree.add_node(agent.get_id(),
                                # store agent
                                agent=agent,
                                name=agent.get_name())

            # link it to the one
            if agent is not self._one:
                self._tree.add_edge(self._one.get_id(), agent.get_id())

        elif isinstance(agent, Iterable):
            for each in agent:
                self.add(each)

        # purified
        return agent

    def link(self, parent, child):
        """
        link two agents

        parameters
        ----------
        child : Agent
            agent child instance
        parent : Agent
            agent parent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("linking agents is not supported before calling Environment.__init__()")

        if isinstance(parent, Agent) and isinstance(child, Agent):
            # clear cache
            self._clear_leaders_cache(child)
            self._clear_leaders_cache(parent)

            # add it, if not already here
            if child.get_id() not in self._tree:
                self.add(child)

            # link
            if not self._tree.has_edge(parent.get_id(), child.get_id()):
                self._tree.add_edge(parent.get_id(), child.get_id())

                # unlink it from the one
                if self._tree.has_edge(self._one.get_id(), child.get_id()) \
                        and self._one.get_id() is not parent.get_id():
                    self._tree.remove_edge(self._one.get_id(), child.get_id())

        elif isinstance(parent, Iterable):
            for each in parent:
                self.link(each, child)

        elif isinstance(child, Iterable):
            for each in child:
                self.link(parent, each)

        return child

    def unlink(self, parent, child):
        """
        unlink two agents

        parameters
        ----------
        child : Agent
            agent child instance
        parent : Agent
            agent parent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("linking agents is not supported before calling Environment.__init__()")

        if isinstance(parent, Agent) and isinstance(child, Agent):
            # clear cache
            self._clear_leaders_cache(child)
            self._clear_leaders_cache(parent)

            # unlink
            if self._tree.has_edge(parent.get_id(), child.get_id()):
                # detach agents
                self._tree.remove_edge(parent.get_id(), child.get_id())

                # check if is a parentless child
                if len(self._tree.in_edges(child.get_id())) == 0:
                    for grand in self.get_parents(parent):
                        self._tree.add_edge(grand.get_id(), child.get_id())

        elif isinstance(parent, Iterable):
            for each in parent:
                self.unlink(each, child)

        elif isinstance(child, Iterable):
            for each in child:
                self.unlink(parent, each)

        return child

    def remove(self, agent):
        """
        remove agent from the tree

        parameters
        ----------
        agent : Agent
            agent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("linking agents is not supported before calling Environment.__init__()")

        # get agent's ID
        agent_id = agent.get_id()

        # check agent
        if agent_id in self._tree:
            # clear cache
            self._clear_leaders_cache(agent)

            # unlink children
            for child in self.get_children(agent):
                self.remove(child)

            # remove it from tree
            self._tree.remove_node(agent_id)

        # remove periodic signals for this agent
        periods = list(self._flow_cycle.keys())
        for period in periods:
            # remove flow
            if agent_id in self._flow_cycle[period]:
                del self._flow_cycle[period][agent_id]

            # no need to keep this period
            if len(self._flow_cycle[period]) == 0:
                del self._flow_cycle[period]

        # recalculate cycle
        if len(self._flow_cycle) > 0:
            self._flow_cycle_period = max(self._flow_cycle.keys())

    def get_children(self, agent, condition=None):
        """ current agent's children

        returns
        -------
        children : Iterable of Agent
            agent's children
        """

        # get agent's ID
        agent_id = agent.get_id()

        # check agent
        if agent_id not in self._tree:
            raise EnvironmentError("agent " + str(agent) + " is not on this environment")

        if condition is None:
            children = [self._tree.nodes[child][Environment.AGENT]
                        for child in self._tree.successors(agent_id)]
        else:
            children = [self._tree.nodes[child][Environment.AGENT]
                        for child in self._tree.successors(agent_id)
                        if condition(self._tree.nodes[child][Environment.AGENT])]

        return children

    def get_parents(self, agent, condition=None):
        """ current agent's parents

        returns
        -------
        parents : Iterable of Agent
            agent's parents
        """

        # get agent's ID
        agent_id = agent.get_id()

        # check agent
        if agent_id not in self._tree:
            raise EnvironmentError("agent " + str(agent) + " is not on this environment")

        if condition is None:
            parents = [self._tree.nodes[parent][Environment.AGENT]
                       for parent in self._tree.predecessors(agent_id)
                       if parent is not agent_id]
        else:
            parents = [self._tree.nodes[parent][Environment.AGENT]
                       for parent in self._tree.predecessors(agent_id)
                       if condition(self._tree.nodes[parent][Environment.AGENT])
                       and parent is not agent_id]

        return parents

    def get_descendants(self, agent, condition=None):
        """ current agent's descendants

        returns
        -------
        descendants : Iterable of Agent
            agent's descendants
        """

        # get agent's ID
        agent_id = agent.get_id()

        # check agent
        if agent_id not in self._tree:
            raise EnvironmentError("agent " + str(agent) + " is not on this environment")

        if condition is None:
            descendants = [self._tree.nodes[descendant][Environment.AGENT]
                           for descendant in self._get_descendants(agent_id)
                           if descendant is not agent_id]
        else:
            descendants = [self._tree.nodes[descendant][Environment.AGENT]
                           for descendant in self._get_descendants(agent_id)
                           if descendant is not agent_id and
                           condition(self._tree.nodes[descendant][Environment.AGENT])]

        return descendants

    def get_leaders(self, agent, condition=None):
        """ current agent's leaders

        returns
        -------
        leaders : Iterable of Agent
            agent's leaders
        """

        # get agent's ID
        agent_id = agent.get_id()

        # check agent
        if agent_id not in self._tree:
            raise EnvironmentError("agent " + str(agent) + " is not on this environment")

        if condition is None:
            leaders = [self._tree.nodes[leader][Environment.AGENT]
                       for leader in self._get_leaders(agent_id)]
        else:
            leaders = [self._tree.nodes[leader][Environment.AGENT]
                       for leader in self._get_leaders(agent_id)
                       if condition(self._tree.nodes[leader][Environment.AGENT])]

        return leaders

    def get_agents(self, condition=None):
        """ get agents

        returns
        -------
        agents : Iterable of Agent
            all agents in this environment
        """

        if condition is None:
            agents = [self._tree.nodes[each][Environment.AGENT] for each in self._tree.nodes]
        else:
            agents = [self._tree.nodes[each][Environment.AGENT] for each in self._tree.nodes
                      if condition(self._tree.nodes[each][Environment.AGENT])]

        return agents

    def get_agent(self, agent_id):
        """ get agent from ID

        returns
        -------
        agent : Agent
            agent
        """
        return self._resolve_id(agent_id)

    def exists(self, agent_id):
        """ agent's existence

        returns
        -------
        bool : existence
            agent's existence in the digital realm
        """

        # existence in digital plane
        return self._tree.has_node(agent_id)

    def plot_agents(self, **kwargs):
        """
        plot environment (agent's hierarchy)
        """

        # plot
        basename = nx.get_node_attributes(self._tree, "name")

        # draw network
        nx.draw_networkx(self._tree, node_size=45, labels=basename, width=0.3,
                         arrowsize=5, font_size=4, **kwargs)

    def post(self, signal, agent=None, period=0):
        """
        post a signal to an agent

        parameters
        ----------
        signal : Signal
            signal
        agent : Agent
            agent's where signal is getting posted
        period : int
            signal period
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("linking agents is not supported before calling Environment.__init__()")

        # get agent id
        agent_id = self._get_agent_id(agent)
        if agent_id is None:
            # post on one's flow
            agent_id = self._one.get_id()

        if signal.get_source():
            # signal is routed as a standard flow
            self._output_flow_from_agent(signal.get_source(), signal)
        else:
            # one is the source
            setattr(signal, Signal.SOURCE, self._one)

            if callable(signal.get_target()):
                # post multicast flow to the agent queue
                self._input_flow_to_one(Environment.MULTICAST, signal, agent_id, priority=1)
            else:
                # unicast flow to the agent queue
                self._input_flow_to_one(Environment.UNICAST, signal, agent_id, priority=1)

        # periodic signals
        if period > 0:
            # flow cycle period
            if period not in self._flow_cycle:
                self._flow_cycle[period] = dict()

            # store agent's signal
            if agent_id not in self._flow_cycle[period]:
                self._flow_cycle[period][agent_id] = [signal]
            else:
                self._flow_cycle[period][agent_id].append(signal)

            # update cycle period
            if period > self._flow_cycle_period:
                self._flow_cycle_period = period

    def plot_flow(self, **kwargs):
        """
        plot signal flow
        """

        # plot
        basename = nx.get_node_attributes(self._tree, "name")
        basename = {agent: basename[agent] for agent in self._flow.nodes if agent in self._tree}

        # draw network
        nx.draw_networkx(self._flow, node_size=45, labels=basename, width=0.3,
                         arrowsize=5, font_size=4, **kwargs)

    # @profile
    def run(self):
        """
        run all agents
        """
        # beginning of execution
        start = time.process_time_ns()

        # periodic flow
        if self._flow_cycle_period > 0:
            # cyclic flow period
            period = (self._time + 1) % self._flow_cycle_period
            if not period:
                period = self._flow_cycle_period

            # get flow, if any
            if period in self._flow_cycle:
                for agent_id in self._flow_cycle[period]:
                    for signal in self._flow_cycle[period][agent_id]:
                        self.post(signal, agent_id)

        # prepare executor
        executor = Executor(environment=self)

        # cascade execution of the one
        executor.run_from(self._one)

        # join executor flow
        self._flow = executor.get_flow()

        # calculate schwifties remainder
        total_schwifties = time.process_time_ns() - start
        agents_schwifties = sum([a.schwifty() for a in self.get_agents()])

        # one schwifties
        assert total_schwifties >= agents_schwifties
        one_schwifties = total_schwifties - agents_schwifties

        # yep
        self._one.add_schwifties(one_schwifties)
        setattr(self._one, Agent.SCHWIFTY, self._one.schwifty() + one_schwifties)

        # update environmental time
        self._time += 1

    def _get_descendants(self, agent_id, stack=None):
        if stack is None:
            # setup function state
            stack = set()
            should_return = True
        else:
            should_return = False

        # get descendants
        for child in self._tree.successors(agent_id):
            if child not in stack:
                stack.add(child)
                self._get_descendants(child, stack)

        if should_return:
            return stack

    # @profile
    def _get_leaders(self, agent_id):
        # check cache
        if agent_id in self._cache_agent_leaders:
            return self._cache_agent_leaders[agent_id]

        # prepare
        leaders = set()

        # leaders trip, it's just a kiss away
        for each_id in filter(lambda parent_id: parent_id is not agent_id, self._tree.predecessors(agent_id)):
            # eye it
            parent = self._tree.nodes[each_id][Environment.AGENT]
            if parent.is_leader():
                leaders.add(parent.get_id())

        # self or not
        if len(leaders) == 0:
            for each_id in \
                    filter(lambda parent_id: parent_id is not agent_id, self._tree.predecessors(agent_id)):
                leaders.update(self._get_leaders(each_id))

        # update cache
        self._cache_agent_leaders[agent_id] = leaders

        # who are we
        return leaders

    def _has_flow(self, agent_id, flow):
        return agent_id in self._flow and \
               flow in self._flow.nodes[agent_id]

    def _get_flow(self, agent_id, flow):
        return self._flow.nodes[agent_id][flow]

    def _join_flow(self, environment):
        """
        join flow from another environment

        parameters
        ----------
        environment : Environment
            the other (environment)
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("joining environments is not supported before calling Environment.__init__()")

        # combine flow
        for node in environment._flow.nodes:
            if node not in self._flow:
                self._flow.add_node(node, **environment._flow.nodes[node])
            else:
                # unicast flow
                if Environment.UNICAST in environment._flow.nodes[node]:
                    unicast = environment._flow.nodes[node][Environment.UNICAST]

                    # bring it here
                    if Environment.UNICAST in self._flow.nodes[node]:
                        self._flow.nodes[node][Environment.UNICAST].add(unicast)
                    else:
                        nx.set_node_attributes(self._flow, {node: {Environment.UNICAST: unicast}})

                # multicast flow
                if Environment.MULTICAST in environment._flow.nodes[node]:
                    unicast = environment._flow.nodes[node][Environment.MULTICAST]

                    # bring it here
                    if Environment.MULTICAST in self._flow.nodes[node]:
                        self._flow.nodes[node][Environment.MULTICAST].add(unicast)
                    else:
                        nx.set_node_attributes(self._flow, {node: {Environment.MULTICAST: unicast}})

        # flow patterns
        self._flow.add_edges_from(environment._flow.edges)

    def _output_flow_from_agent(self, agent, obj):
        # setup signal's attributes
        if isinstance(obj, Signal):
            # setup source
            if not obj.get_source():
                setattr(obj, Signal.SOURCE, agent)

        if isinstance(obj, Iterable):
            for each in obj:
                self._output_flow_from_agent(agent, each)

        elif isinstance(obj, Kernel):
            # run kernel object on agent
            start = time.process_time_ns()
            output = obj(agent)
            schwifties = time.process_time_ns() - start

            # get schwifty
            agent.add_schwifties(schwifties)

            # scream
            if Environment.VERBOSE:
                print("[  ---->] running signal " + type(obj).__name__ +
                      "@" + str(obj.get_source()) + " - " + str(schwifties))

            if isinstance(output, Iterable):
                for signals in output:
                    self._output_flow_from_agent(agent, signals)

        elif isinstance(obj, Purple):
            # collect purple signal
            self._input_flow_to_one(Environment.UNICAST, obj)

        elif isinstance(obj, Signal):
            # get signal target
            target = obj.get_target()

            if callable(target) or target is None:
                # multicast input signal
                self._input_flow_to_us(agent.get_id(), obj)

            else:
                # not callable targets
                if agent.get_id() not in self._flow:
                    self._flow.add_node(agent.get_id(), unicast=SignalQueue())

                # unicast flow
                if Environment.UNICAST not in self._flow.nodes[agent.get_id()]:
                    nx.set_node_attributes(self._flow,
                                           {agent.get_id(): {Environment.UNICAST: SignalQueue()}})

                # propagate input signal to target
                self._input_flow_to_target(agent.get_id(), target, obj)

    def _input_flow_to_one(self, flow_type, obj, agent_id=None, priority=0):
        if not obj.get_source():
            setattr(obj, Signal.SOURCE, self._one)

        # set environment on purple call
        setattr(obj, Purple.ENVIRONMENT, self)

        # set target
        target_id = agent_id
        if target_id is None:
            target_id = self._one.get_id()

        # one purple flow
        if target_id not in self._flow:
            self._flow.add_node(target_id, unicast=SignalQueue(), multicast=SignalQueue())

        # multicast traffic
        if Environment.MULTICAST not in self._flow.nodes[target_id]:
            nx.set_node_attributes(self._flow, {target_id: {Environment.MULTICAST: SignalQueue()}})

        # unicast traffic
        if Environment.UNICAST not in self._flow.nodes[target_id]:
            nx.set_node_attributes(self._flow, {target_id: {Environment.UNICAST: SignalQueue()}})

        # add flow
        self._flow.nodes[target_id][flow_type].append(obj, priority)

    def _input_flow_to_target(self, source_id, target_id, obj):
        if isinstance(target_id, Iterable):
            for each_id in target_id:
                # propagate input signal to each target
                self._input_flow_to_target(source_id, each_id, obj)
        else:
            # create target input flow
            if target_id not in self._flow:
                self._flow.add_node(target_id, unicast=SignalQueue())

            if Environment.UNICAST not in self._flow.nodes[target_id]:
                nx.set_node_attributes(self._flow, {target_id: {Environment.UNICAST: SignalQueue()}})

            # add signal to the flow
            self._flow.add_edge(source_id, target_id)
            self._flow.nodes[target_id][Environment.UNICAST].append(obj, obj.get_priority())

    def _input_flow_to_us(self, source_id, obj):
        # get parents
        leaders = self._get_leaders(source_id)

        # create target input flow
        for leader_id in leaders:
            # multicast queue
            if leader_id not in self._flow:
                self._flow.add_node(leader_id, multicast=SignalQueue())

            if Environment.MULTICAST not in self._flow.nodes[leader_id]:
                nx.set_node_attributes(self._flow, {leader_id: {Environment.MULTICAST: SignalQueue()}})

            # add signal to the flow
            self._flow.nodes[leader_id][Environment.MULTICAST].append(obj, obj.get_priority())

    def _resolve_id(self, agent_id):
        # are you here agent ?
        if agent_id not in self._tree:
            raise EnvironmentError("no route to agent")

        return self._tree.nodes[agent_id][Environment.AGENT]

    @staticmethod
    def _get_agent_id(agent):
        if hasattr(agent, Agent.GET_ID):
            return agent.get_id()
        return agent

    @staticmethod
    def _has_call_method(agent):
        return "__call__" in agent.__class__.__dict__.keys()

    def _clear_leaders_cache(self, agent):
        # no cache for you
        if agent.get_id() in self._cache_agent_leaders:
            del self._cache_agent_leaders[agent.get_id()]

            # clean child cache
            for child in self.get_children(agent):
                self._clear_leaders_cache(child)

    # @profile
    def _is_agent_runnable(self, agent):
        leaders = self._get_leaders(agent.get_id())

        return any([self._has_flow(leader_id, Environment.MULTICAST) for leader_id in leaders]) or \
               self._has_flow(agent.get_id(), Environment.UNICAST) or callable(agent)


class Executor(Environment):
    def __init__(self, environment):
        # store instance to the real environment
        self._environment = environment
        assert isinstance(self._environment, Environment)

        # store tree's reference
        self._tree = self._environment._tree

        # signal's tree flow
        self._flow = nx.DiGraph()

        # initial time
        self._time = self._environment.get_time()

        # init done
        self._init_done = True

        # propagate the one
        self._one = self._environment._one

        # periodic flows
        self._flow_cycle = self._environment._flow_cycle
        self._flow_cycle_period = self._environment._flow_cycle_period

    def add(self, agent):
        """
        add a new agent to this environment

        parameters
        ----------
        agent : Agent
            agent_instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("adding agents is not supported before calling Environment.__init__()")

        assert isinstance(self._environment, Environment)

        if isinstance(agent, Agent):
            self._environment._input_flow_to_one(Environment.UNICAST, Add(agent))

        elif isinstance(agent, Iterable):
            for each in agent:
                self.add(each)

        # purified
        return agent

    def link(self, parent, child):
        """
        link two agents

        parameters
        ----------
        child : Agent
            agent child instance
        parent : Agent
            agent parent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("linking agents is not supported before calling Environment.__init__()")

        assert isinstance(self._environment, Environment)

        if isinstance(parent, Agent) and isinstance(child, Agent):
            self._environment._input_flow_to_one(Environment.UNICAST, Link(parent, child))

        elif isinstance(parent, Iterable):
            for each in parent:
                self.link(each, child)

        elif isinstance(child, Iterable):
            for each in child:
                self.link(parent, each)

        return child

    def unlink(self, parent, child):
        """
        unlink two agents

        parameters
        ----------
        child : Agent
            agent child instance
        parent : Agent
            agent parent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("unlinking agents is not supported before calling Environment.__init__()")

        assert isinstance(self._environment, Environment)

        if isinstance(parent, Agent) and isinstance(child, Agent):
            self._environment._input_flow_to_one(Environment.UNICAST, Unlink(parent, child))

        elif isinstance(parent, Iterable):
            for each in parent:
                self.unlink(each, child)

        elif isinstance(child, Iterable):
            for each in child:
                self.unlink(parent, each)

        return child

    def remove(self, agent):
        """
        remove agent from the tree

        parameters
        ----------
        agent : Agent
            agent instance
        """

        if not hasattr(self, "_init_done"):
            raise EnvironmentError("removing agents is not supported before calling Environment.__init__()")

        assert isinstance(self._environment, Environment)

        self._environment._input_flow_to_one(Environment.UNICAST, Remove(agent))

    # @profile
    def run_one(self, agent):
        """
        execute an agent (only for environmental executors)

        parameters
        ----------
        agent : Agent
            agent instance
        """

        if self._is_agent_runnable(agent):
            # set temporary environment reference
            setattr(agent, Agent.ENVIRONMENT, self)

            # get agent's flow
            flow = InteractionQueue(agent)

            # get parents
            leaders = self._get_leaders(agent.get_id())

            # agent's flow
            if self._has_flow(agent.get_id(), Environment.UNICAST):
                flow.add(self._get_flow(agent.get_id(), Environment.UNICAST))

            # combine leader's multicast flow
            for leader_id in leaders:
                if self._has_flow(leader_id, Environment.MULTICAST):
                    flow.add(self._get_flow(leader_id, Environment.MULTICAST))

            # execute precursor flow
            if InteractionQueue.PRECURSOR in flow:
                for rest in flow[InteractionQueue.PRECURSOR]:
                    assert rest is rest and flow

            # agent sanity check
            if not agent.get_environment():
                raise EnvironmentError("can't execute agent without environment")

            # run agent, and distribute signals
            if callable(agent):
                start = time.process_time_ns()
                output = agent(flow)
                schwifties = time.process_time_ns() - start

                # get schwifty
                agent.add_schwifties(schwifties)

                # scream
                if Environment.VERBOSE:
                    print("[---->] running agent " + str(agent) + " - " + str(schwifties))

                if isinstance(output, Iterable):
                    for signals in output:
                        self._output_flow_from_agent(agent, signals)
            else:
                # booyah
                schwifties = 0

            # execute priority flow
            if InteractionQueue.PRIORITY in flow:
                for rest in flow[InteractionQueue.PRIORITY]:
                    assert rest is rest and flow

            # execute remaining flow, if any
            for rest in flow:
                assert rest is rest and flow

            # shelter it
            assert not len(flow)

            # and interact
            for interaction in flow.interactions:
                self._output_flow_from_agent(agent, interaction)

            # restore environment reference
            setattr(agent, Agent.ENVIRONMENT, self._environment)

        else:
            # booyah
            schwifties = 0

        # sync time
        setattr(agent, Agent.TIME, self._time + 1)
        setattr(agent, Agent.SCHWIFTY, schwifties)

    # @profile
    def run_from(self, agent):
        """
        cascade execution of agents

        parameters
        ----------
        agent : Agent
            agent instance
        """

        # run children
        for child in self.get_children(agent):
            # there must be a way out of here
            if child.get_time() <= self._time:
                self.run_from(child)

        self.run_one(agent)

    def run(self):
        raise EnvironmentError("can't run executor")

    def _input_flow_to_one(self, flow_type, obj, agent_id=None, priority=0):
        # purple flow goes directly to the real environment
        assert isinstance(self._environment, Environment)
        self._environment._input_flow_to_one(flow_type, obj, agent_id, priority)

    def _has_flow(self, agent_id, flow):
        # propagate flow from the real environment
        assert isinstance(self._environment, Environment)
        return self._environment._has_flow(agent_id, flow)

    def _get_flow(self, agent_id, flow):
        # propagate flow from the real environment
        assert isinstance(self._environment, Environment)
        return self._environment._get_flow(agent_id, flow)

    def _get_leaders(self, agent_id):
        # propagate leaders from the real environment
        assert isinstance(self._environment, Environment)
        return self._environment._get_leaders(agent_id)
