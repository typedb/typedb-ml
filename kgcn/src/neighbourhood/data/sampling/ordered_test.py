
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
        self.assertListEqual(list(first_n), self._p[:n])

    def test_when_sample_size_more_than_population(self):
        ns = [7, 10, 11, 15, 24]

        for n in ns:
            with self.subTest(f'population size = {len(self._p)}, n = {n}'):
                population = _gen(self._p)

                first_n = ordered.ordered_sample(population, n)

                self.assertListEqual(list(first_n), list(_gen(self._p))[:n])

    def test_when_sample_size_is_zero(self):

        population = _gen(self._p)
        n = 0
        first_n = ordered.ordered_sample(population, n)
        self.assertListEqual(list(first_n), [])

    def test_when_population_size_is_zero(self):

        population = _gen([])
        n = 5
        first_n = ordered.ordered_sample(population, n)
        self.assertRaises(ValueError)


if __name__ == "__main__":
    unittest.main()
