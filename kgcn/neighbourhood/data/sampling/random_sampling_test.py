import unittest

import kgcn.neighbourhood.data.sampling.random_sampling as rnd


def _gen(l):
    for i in l:
        yield i


class TestRandomSample(unittest.TestCase):
    def setUp(self):
        self._p = list(range(5))

    def test_when_sample_size_less_than_population(self):

        population = _gen(self._p)
        n = 3
        rnd_samples = rnd.random_sample(population, n)
        self.assertEqual(min(n, len(self._p)), len(rnd_samples))
        self.assertTrue(set(rnd_samples).issubset(set(self._p)))

    def test_when_sample_size_more_than_population(self):
        ns = [7, 10, 11, 15, 24]

        for n in ns:
            with self.subTest(f'population size = {len(self._p)}, n = {n}'):
                population = _gen(self._p)

                rnd_samples = rnd.random_sample(population, n)
                self.assertEqual(min(n, len(self._p)), len(rnd_samples))
                self.assertTrue(set(rnd_samples).issubset(set(self._p)))

    def test_when_sample_size_is_zero(self):

        population = _gen(self._p)
        n = 0
        rnd_samples = rnd.random_sample(population, n)
        self.assertListEqual(list(rnd_samples), [])

    def test_when_population_size_is_zero(self):

        population = _gen([])
        n = 5
        with self.assertRaises(ValueError) as context:
            first_n = rnd.random_sample(population, n)

        self.assertTrue('Population is empty' in str(context.exception))
