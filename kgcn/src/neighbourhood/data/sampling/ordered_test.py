
import unittest

import kgcn.src.neighbourhood.data.sampling.ordered as ordered


def _gen(l):
    for i in l:
        yield i


class TestOrderedSample(unittest.TestCase):
    def setUp(self):
        self._p = list(range(5))

    def test_when_sample_size_less_than_population(self):

        population = _gen(self._p)
        n = 3
        first_n = ordered.ordered_sample(population, n)
        self.assertListEqual(first_n, self._p[:n])

    def test_when_sample_size_more_than_population(self):
        ns = [7, 10, 11, 15, 24]

        expanded_population = []
        for i in range(5):
            expanded_population += self._p

        for n in ns:
            with self.subTest(f'population size = {len(self._p)}, n = {n}'):
                population = _gen(self._p)

                first_n = ordered.ordered_sample(population, n)

                self.assertListEqual(first_n, expanded_population[:n])

    def test_when_sample_size_is_zero(self):

        population = _gen(self._p)
        n = 0
        first_n = ordered.ordered_sample(population, n)
        self.assertListEqual(first_n, [])
