from warpz.kernel.signals import *


class Kernel(Signal):
    """"
    kernel signaling (calls executed on the agent immediately after run)

    """

    def __init__(self, **kwargs):
        # initialize signal
        super().__init__(**kwargs)

    def __call__(self, agent):
        yield


class Purple(Signal):
    """"
    purple signaling (calls executed by the one after and running the environment)

    """
    ENVIRONMENT = "_environment"

    def __init__(self, **kwargs):
        # initialize signal
        super().__init__(**kwargs)
        self._environment = None

    def get_environment(self):
        return self._environment

    def __call__(self, one):
        yield


""""
  purple calls

"""


class Add(Purple):
    def __init__(self, agent, **kwargs):
        self.agent = agent
        super().__init__(**kwargs)

    def __call__(self, one):
        self.get_environment().add(self.agent)


class Link(Purple):
    def __init__(self, parent, child, **kwargs):
        self.parent = parent
        self.child = child
        super().__init__(**kwargs)

    def __call__(self, one):
        self.get_environment().link(self.parent, self.child)


class Unlink(Purple):
    def __init__(self, parent, child, **kwargs):
        self.parent = parent
        self.child = child
        super().__init__(**kwargs)

    def __call__(self, one):
        self.get_environment().unlink(self.parent, self.child)


class Remove(Purple):
    def __init__(self, agent, **kwargs):
        self.agent = agent
        super().__init__(**kwargs)

    def __call__(self, one):
        self.get_environment().remove(self.agent)


""""
  kernel calls

"""


class Gone(Kernel):
    def __init__(self):
        super().__init__()

    def __call__(self, agent):
        agent.remove()


class Born(Kernel):
    def __init__(self, new, **kwargs):
        self.new = new
        super().__init__(**kwargs)

    def __call__(self, agent):
        for parent in agent.get_parents():
            parent.link(self.new)

