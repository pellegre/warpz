import unittest
from warpz.kernel.environment import *
from warpz.space.box import *

import matplotlib.pyplot as plt


class TP(Particle):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, *args, **kwargs):
        yield Transport(0.02, [1.0, 0.0])


class TestCore(unittest.TestCase):

    def test_box_instance(self):
        box = Box(grid=10)

        particle = box.put(Particle(position=[0.95, 0.95]))
        self.assertEqual(particle.get_cell().index, (9, 9))

        particle = box.put(Particle(position=[0.00, 0.00]))
        self.assertEqual(particle.get_cell().index, (0, 0))

        particle = box.put(Particle(position=[1.00, 1.00]))
        self.assertEqual(particle.get_cell().index, (0, 0))

        particle = box.put(Particle(position=[1.05, 1.05]))
        self.assertEqual(particle.get_cell().index, (0, 0))

        particle = box.put(Particle(position=[1.15, 1.15]))
        self.assertEqual(particle.get_cell().index, (1, 1))

        box.run()

    def test_box_transport(self):
        box = Box(grid=10)
        particle = box.put(TP(position=[0.51, 0.51]))
        self.assertEqual(particle.get_cell().index, (5, 5))

        self.assertEqual(len(box.get_children()), 100)

        for i in range(0, 20):
            box.run()

            value = int(10 * particle.get_position()[0])
            self.assertEqual(value, particle.get_cell().index[0])


if __name__ == '__main__':
    unittest.main()
