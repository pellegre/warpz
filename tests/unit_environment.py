import unittest
from warpz.kernel.environment import *
from warpz.kernel.signals import *
from warpz.kernel.universe import *

import matplotlib.pyplot as plt

from enum import IntEnum


class TestCore(unittest.TestCase):

    def test_environmental_instance(self):
        universe = Universe()

        a = universe.add(Agent(name="A"))

        ba = a.link(Agent(name="BA"))
        bb = a.link(Agent(name="BB"))

        ca = bb.link(Agent(name="CA"))
        cb = bb.link(Agent(name="CB"))
        cc = bb.link(Agent(name="CC"))
        cd = bb.link(Agent(name="CD"))

        ba.link(cc)
        ba.link(cd)
        ce = ba.link(Agent(name="CE"))

        a_children = a.get_children()
        self.assertEqual(a_children[0], ba)
        self.assertEqual(a_children[1], bb)

        a_children = a.get_children(condition=lambda agent: "BA" in agent.get_name())
        self.assertEqual(a_children[0], ba)

        c_agents = universe.get_agents(condition=lambda agent: "C" in agent.get_name())
        self.assertEqual(len(c_agents), 5)

        cd_agent = bb.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(cd_agent[0], cd)

        cd_agent = ba.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(cd_agent[0], cd)

        cd_agent = a.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(len(cd_agent), 0)

        bb.unlink(cd)

        cd_agent = bb.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(len(cd_agent), 0)

        cd_agent = ba.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(cd_agent[0], cd)

        ba.unlink(cd)

        cd_agent = ba.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(len(cd_agent), 0)

        cd_agent = a.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(cd_agent[0], cd)

        a.unlink(cd)

        cd_agent = a.get_children(condition=lambda agent: "CD" in agent.get_name())
        self.assertEqual(len(cd_agent), 0)

    class Z(IntEnum):
        A = 1
        B = 2
        C = 3
        D = 4
        E = 5

    class W(Agent):
        def __init__(self, send=None, extra=None, **kwargs):
            self.send, self.extra, self.got = send, extra, None
            super(TestCore.W, self).__init__(**kwargs)

        def __call__(self, signals):
            if self.extra is not None and self.extra in signals:
                self.got = list(signals[self.extra])
            else:
                self.got = list(signals)

            send = self.send
            self.send = None

            if "CA" in self.get_name():
                if self.extra in signals:
                    print(self.get_name(), self.get_time(),
                          "executed", send, *signals._signal_stack[self.extra.get_id()])

            yield send

    def test_environmental_directed_signals(self):
        env = Environment()

        aa = env.add(TestCore.W(name="AA"))
        ab = env.add(TestCore.W(name="AB"))
        ac = env.add(TestCore.W(name="AC"))
        ad = env.add(TestCore.W(name="AD"))

        bb = env.add(TestCore.W(name="BB"))
        bc = env.add(TestCore.W(name="BC"))
        bd = env.add(TestCore.W(name="BD"))

        ba = env.add(TestCore.W(name="BA", extra=bd))

        ca = env.add(TestCore.W(name="CA"))
        cb = env.add(TestCore.W(name="CB"))
        cc = env.add(TestCore.W(name="CC"))
        cd = env.add(TestCore.W(name="CD"))
        ce = env.add(TestCore.W(name="CE"))

        da = env.add(TestCore.W(name="DA"))

        env.link(aa, ba)
        env.link(aa, bb)
        env.link(aa, ca)

        env.link(ab, ac)
        env.link(ab, bc)
        env.link(ab, cb)

        env.link(ac, bd)

        env.link(ba, cc)
        env.link(ba, cd)

        env.link(cc, ce)
        env.link(cc, ad)
        env.link(cc, da)

        bd.send = [Signal(TestCore.Z.B, bc), Signal(TestCore.Z.D, target=ba)]
        ba.send = Signal(TestCore.Z.A, target=aa)

        env.run()

        cd.send = [Signal(TestCore.Z.D, target=ce)]
        ab.send = Signal(lambda a: a.get_id(), target=aa)

        env.run()

        self.assertEqual(len(aa.got), 1)
        self.assertEqual(aa.got[0], TestCore.Z.A)

        self.assertEqual(len(ba.got), 1)
        self.assertEqual(ba.got[0], TestCore.Z.D)

        env.run()

        self.assertEqual(len(aa.got), 1)
        self.assertEqual(aa.got[0], aa.get_id())

        za = TestCore.W(name="ZA")
        zb = TestCore.W(name="ZB")
        cd.send = [Born(new=za)]
        ab.send = Signal(lambda a: Born(new=zb), target=aa)

        env.run()

        ba.send = Signal(TestCore.Z.D, target=za)

        env.run()

        # env.plot_agents()
        # plt.show()

        za_agent = env.get_children(ba, condition=lambda agent: "ZA" in agent.get_name())
        self.assertEqual(za_agent[0], za)

        ba.send = Signal(TestCore.Z.A, target=[za, cc, aa])

        env.run()

        self.assertEqual(len(za.got), 1)
        self.assertEqual(za.got[0], TestCore.Z.D)

        env.run()

        self.assertEqual(len(za.got), 1)
        self.assertEqual(za.got[0], TestCore.Z.A)

        self.assertEqual(len(aa.got), 1)
        self.assertEqual(aa.got[0], TestCore.Z.A)

        for each in env.get_agents():
            self.assertTrue(each.get_environment() is env)

    class Counter(Agent):
        def __init__(self, signal):
            self.signal_counter, self.other_agents = {}, []
            self.signal = signal
            super().__init__()

        def __call__(self, signals):
            # count signals
            for z in signals:
                if z not in self.signal_counter:
                    self.signal_counter[z] = 1
                else:
                    self.signal_counter[z] += 1

            # send to others
            yield [Signal(self.signal, target=self.other_agents)]

    @staticmethod
    def build_counter_cluster(n, signal_type):
        cluster = []
        for i in range(0, n):
            cluster.append(TestCore.Counter(signal_type))

        return cluster

    @staticmethod
    def link_all(agents, to):
        for a in agents:
            a.other_agents = to(a)

    def test_environmental_directed_broadcast(self):
        universe = Universe()
        env = universe.get_environment()

        # agents
        n = 5

        # cluster (A)
        top_a = universe.add(TestCore.W(name="TOP_A"))

        cluster_a = self.build_counter_cluster(n, TestCore.Z.A)
        env.link(top_a, cluster_a)

        # cluster (B)
        top_b = env.add(TestCore.W(name="TOP_B"))

        cluster_b = self.build_counter_cluster(n, TestCore.Z.B)
        env.link(top_b, cluster_b)
        env.link(cluster_a[0], top_b)

        # cluster (C)
        top_c = universe.add(TestCore.W(name="TOP_C"))

        cluster_c = self.build_counter_cluster(n, TestCore.Z.C)
        env.link(top_c, cluster_c)

        # cluster (D)
        top_d = env.add(TestCore.W(name="TOP_D"))

        cluster_d = self.build_counter_cluster(n, TestCore.Z.D)
        env.link(top_d, cluster_d)
        env.link(cluster_c[1], top_d)

        # env.plot_agents()
        # plt.show()

        # link all
        clusters = cluster_a + cluster_b + cluster_c + cluster_d
        self.link_all(clusters, lambda a: clusters)

        env.run()

        # n
        env.run()

        # should be equal to n
        for i, agent in zip(range(0, n), clusters):
            for z in agent.signal_counter:
                self.assertEqual(agent.signal_counter[z], n)

        self.link_all(clusters, lambda a: Signal.BROADCAST(lambda b: hasattr(b, "signal") and (b.signal == a.signal)))

        # 2n, clean last signals
        env.run()

        # 3n, should be multicasted by the one
        env.run()

        # should be equal to n
        signal_set = [TestCore.Z.A, TestCore.Z.B, TestCore.Z.C, TestCore.Z.D]
        cluster_set = [cluster_a, cluster_b, cluster_c, cluster_d]

        for i, cluster in enumerate(cluster_set):
            for agent in cluster_set[i]:
                for j in range(0, len(signal_set)):
                    if i == j:
                        self.assertEqual(agent.signal_counter[signal_set[j]], 3 * n)
                    else:
                        self.assertEqual(agent.signal_counter[signal_set[j]], 2 * n)

        # 4n, clean up signals
        env.run()

        setattr(top_a, "_leader", True)
        setattr(top_c, "_leader", True)
        setattr(env, "_cache_agent_leaders", dict())

        # --- propagate to new leaders, it will clean up the flow
        env.run()

        # 5n, communication should be contained within leader's groups
        env.run()

        for i, cluster in enumerate(cluster_set):
            for agent in cluster_set[i]:
                for j in range(0, len(signal_set)):
                    if i == j:
                        self.assertEqual(agent.signal_counter[signal_set[j]], 5 * n)
                    else:
                        self.assertEqual(agent.signal_counter[signal_set[j]], 2 * n)

        for each in env.get_agents():
            self.assertTrue(each.get_environment() is env)

    def check_leader(self, agent, leader, env):
        leaders = list([leader.get_id() for leader in env.get_leaders(agent)])
        if hasattr(leader, "get_id"):
            self.assertTrue(leader.get_id() in leaders)
        else:
            self.assertTrue(leader in leaders)

    def test_environmental_leaders_hierarchy(self):
        universe = Universe()

        aa = universe.add(TestCore.W(name="AA", leader=True))
        ab = universe.add(TestCore.W(name="AB"))

        bb = aa.link(TestCore.W(name="BB", leader=True))

        bc = aa.link(TestCore.W(name="BC"))
        bd = ab.link(TestCore.W(name="BD"))

        ba = ab.link(TestCore.W(name="BA", extra=bd, leader=True))

        cb = bb.link(TestCore.W(name="CB"))
        cc = bb.link(TestCore.W(name="CC"))
        ca = bb.link(TestCore.W(name="CA", extra=cc))

        cd = ba.link(TestCore.W(name="CD"))
        ce = ba.link(TestCore.W(name="CE"))
        cf = ba.link(TestCore.W(name="CF"))
        cg = bd.link(TestCore.W(name="CG"))

        bd.link(ce)
        bd.link(cf)
        bd.link(cg)

        bc.link(cb)
        bc.link(cc)
        ch = bc.link(TestCore.W(name="CH"))

        counter_b = bc.link(TestCore.Counter(TestCore.Z.B))
        counter_c = ce.link(TestCore.Counter(TestCore.Z.C))

        self.check_leader(ca, bb, universe.get_environment())
        self.check_leader(ce, ba, universe.get_environment())
        self.check_leader(bc, aa, universe.get_environment())

        self.check_leader(bb, aa, universe.get_environment())

        one = universe

        self.check_leader(aa, one, universe.get_environment())
        self.check_leader(bd, one, universe.get_environment())
        self.check_leader(ba, one, universe.get_environment())

        universe.run()

        for ag in universe.get_agents():
            self.assertEqual(universe.get_time(), ag.get_time())

        one_leaders = one.get_leaders()
        self.assertEqual(len(one_leaders), 0)

        one_parents = one.get_parents()
        self.assertEqual(len(one_parents), 0)

        for ag in universe.get_agents():
            ag_leaders = ag.get_leaders()
            self.assertTrue(len(ag_leaders) > 0)

        counter_b.other_agents = [ca, cc]

        universe.run()

        counter_b.other_agents = lambda a: "BC" in a.get_name()

        universe.run()
        universe.run()

        self.assertEqual(len(bc.got), 1)
        self.assertEqual(bc.got[0], TestCore.Z.B)

        counter_c.other_agents = lambda a: "CD" in a.get_name() or "CF" in a.get_name()

        universe.run()
        universe.run()

        self.assertEqual(len(cd.got), 1)
        self.assertEqual(cd.got[0], TestCore.Z.C)

        self.assertEqual(len(cf.got), 1)
        self.assertEqual(cf.got[0], TestCore.Z.C)

        counter_c.other_agents = lambda a: "CG" in a.get_name()

        universe.run()
        universe.run()

        self.assertTrue("BA" in universe.get_environment().get_leaders(counter_c)[0].get_name())

        self.assertEqual(len(cg.got), 0)

        universe.get_environment().link(ba, bd)

        universe.get_environment().run()
        universe.get_environment().run()

        self.assertTrue("BA" in universe.get_environment().get_leaders(cg)[0].get_name())
        self.assertEqual(len(cg.got), 1)

        counter_c.other_agents = []
        counter_b.other_agents = []

        universe.get_environment().run()
        universe.get_environment().run()

        bc.extra = ch
        ch.send = Signal(TestCore.Z.D, target=lambda a: "BC" in a.get_name())

        universe.get_environment().run()
        universe.get_environment().run()

        self.assertTrue(universe.get_environment().get_leaders(ch)[0] == universe.get_environment().get_leaders(bc)[0])
        self.assertEqual(bc.got[0], TestCore.Z.D)

        for each in universe.get_environment().get_agents():
            self.assertTrue(each.get_environment() is universe.get_environment())

    class Y(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, signals):
            yield None

    class X(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def __call__(self, signals):
            yield None

    class Touch:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0

        def __call__(self, agent):
            self.count += 1

    class TouchAndPropagate:
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0

        def __call__(self, agent):
            self.count += 1
            if self.count > 50:
                yield Gone()
            else:
                yield [Signal(action=self, target=Signal.BROADCAST())]

    def test_environmental_signal_queue_execution(self):
        universe = Universe()

        aa = universe.add(TestCore.X(name="AA"))
        ab = universe.add(TestCore.Y(name="AB"))

        ba = aa.link(TestCore.Y(name="BA"))
        bb = aa.link(TestCore.Y(name="BB"))

        bc = ab.link(TestCore.X(name="BC"))
        bd = ab.link(TestCore.X(name="BD"))

        touch = TestCore.Touch()
        universe.post(Signal(action=touch, target=ba, source=bd))

        universe.run()
        self.assertEqual(touch.count, 1)

        universe.run()

        touch = TestCore.Touch()

        universe.post(Signal(action=touch, source=bd))

        universe.run()
        universe.run()

        self.assertEqual(touch.count, len(universe.get_agents()))

        touch_and_propagate = TestCore.TouchAndPropagate()
        universe.post(Signal(action=touch_and_propagate, target=ba, source=bd))

        universe.run()

        self.assertEqual(touch_and_propagate.count, 1)

        universe.run()

        self.assertEqual(touch_and_propagate.count, 1 + len(universe.get_agents()))

        universe.run()

        universe.run()
        universe.run()

        last_count = touch_and_propagate.count

        universe.run()
        universe.run()
        universe.run()
        universe.run()

        self.assertEqual(touch_and_propagate.count, last_count)

        self.assertEqual(len(universe.get_agents()), 0)

        # warpz.plot_agents()
        # plt.show()


if __name__ == '__main__':
    unittest.main()
    # test = TestCore()
    # test.test_environmental_directed_signals()
