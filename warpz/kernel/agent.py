import uuid as zz


# base class for a spatial agent
class Agent(object):
    """"
    Agent base class

    attributes
    ----------
    name : str
        user defined agent's name
    position : object
        agent's position in the universe
    leader : bool
        agent's leadership

    methods
    -------
    live()
    """

    LEADER = "_leader"
    TIME = "_time"
    POSITION = "_position"
    ENVIRONMENT = "_environment"
    SCHWIFTY = "_schwifty"

    GET_ID = "get_id"

    def __init__(self, name=None, position=None, leader=False):
        # default agent ID
        self._id = zz.uuid4().int

        # leader agent
        self._leader = leader

        # set agent's name
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

        # agent's name
        self._name += "[" + hex(id(self)) + "]"

        # agent's position
        self._position = position

        # initial time
        self._time = 0

        # environment
        self._environment = None

        # init done
        self._init_done = True

        # accumulated schwifties
        self._schwifties = 0

        # local schwifty
        self._schwifty = 0

    def __str__(self):
        return "< agent ( " + self._name + " ) >"

    def schwifty(self):
        """ get agent's current schwifty

        """

        return self._schwifty

    def get_schwifties(self):
        """ get agent's accumulated schwifties

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

    def get_environment(self):
        """ get underlying environment

        """

        return self._environment

    def get_children(self, condition=None):
        """ agent's children

        returns
        -------
        children : Iterable of Agent
            agent's children
        """

        if self._environment is None:
            raise EnvironmentError("no children yet")

        return self._environment.get_children(self, condition)

    def get_parents(self, condition=None):
        """ current agent's parents

        returns
        -------
        parents : Iterable of Agent
            agent's parents
        """

        if self._environment is None:
            raise EnvironmentError("no parents yet")

        return self._environment.get_parents(self, condition)

    def get_leaders(self, condition=None):
        """ current agent's leaders

        returns
        -------
        leaders : Iterable of Agent
            agent's leaders
        """

        if self._environment is None:
            raise EnvironmentError("no leaders yet")

        return self._environment.get_leaders(self, condition)

    def get_agents(self, condition=None):
        """ agent's descendants

        returns
        -------
        descendants : Iterable of Agent
            agent's descendants
        """

        if self._environment is None:
            raise EnvironmentError("no descendants yet")

        return self._environment.get_descendants(self, condition)

    def exists(self):
        """ agent's existence

        returns
        -------
        bool : existence
            agent's existence in the digital realm
        """

        # existence in digital plane
        return self._environment.exists(self.get_id())

    def is_leader(self):
        """ agent's leadership state

        returns
        -------
        id : int
            agent's leadership state
        """

        return self._leader

    def get_time(self):
        """ agent's current time

        returns
        -------
        time : int
            agent's time
        """

        return self._time

    def get_id(self):
        """ agent's ID

        returns
        -------
        id : int
            agent's ID
        """

        return self._id

    def get_position(self):
        """ agent's position

        returns
        -------
        position : array
            agent's position
        """

        return self._position

    def get_name(self):
        """ agent's name

        returns
        -------
        name : str
            agent's name
        """

        return self._name

    def link(self, child):
        """ link a new child

        returns
        -------
        name : str
            agent's name
        """

        if self._environment is None:
            raise EnvironmentError("no descendants yet")

        return self._environment.link(self, child)

    def unlink(self, child):
        """ unlink a child

        returns
        -------
        name : str
            agent's name
        """

        if self._environment is None:
            raise EnvironmentError("no descendants yet")

        return self._environment.unlink(self, child)

    def remove(self):
        """ link a new child

        returns
        -------
        name : str
            agent's name
        """

        if self._environment is None:
            raise EnvironmentError("no descendants yet")

        return self._environment.remove(self)

