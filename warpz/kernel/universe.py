from warpz.kernel.environment import *


class Universe(Agent):
    def __init__(self, **kwargs):
        super().__init__(leader=True, **kwargs)

        # environment
        self._environment = Environment(one=self)

    def get_agents(self, condition=None):
        """ agent's descendants

        returns
        -------
        descendants : Iterable of Agent
            agent's descendants
        """

        if self._environment is None:
            raise EnvironmentError("no descendants yet")

        if condition is None:
            return self._environment.get_agents(condition=lambda agent: agent is not self)
        else:
            return self._environment.get_agents(condition=lambda agent: condition(agent) and agent is not self)

    def get_count(self):
        """ agent count

        returns
        -------
        count : int
            amount of agents
        """

        return self._environment.get_count()

    def add(self, agent):
        """
        add a new agent to the universe

        parameters
        ----------
        agent : Agent
            agent_instance
        """

        return self.link(agent)

    def post(self, signal, agent=None, period=0):
        """
        post a signal in the universe to an agent

        parameters
        ----------
        signal : Signal
            signal
        agent : Agent
            agent's where signal is getting posted
        period : int
            signal period
        """

        return self._environment.post(signal, agent, period)

    def run(self):
        """
        run all agents
        """

        # run the environment
        self._environment.run()

    def __call__(self, *args, **kwargs):
        """
        callable agent
        """

        yield
