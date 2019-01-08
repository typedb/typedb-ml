import unittest

import kgcn.src.neighbourhood.data.sampling.random as rnd


def generator(iterable):
    for it in iterable:
        yield it


class TestRandomSample(unittest.TestCase):

    def _default_tests(self, population, sample_size, samples):
        """
        These subTests make check that should always be true of the result if there is one or more item in the
        population
        :param population:
        :param sample_size:
        :param samples:
        :return:
        """
        with self.subTest("Number or Samples"):
            self.assertEqual(sample_size, len(samples))

        with self.subTest("Samples belong to Population"):
            self.assertTrue(set(samples) <= set(population))

    def test_sampled_without_replacement(self):
        """
        Check that if population is greater than `sample_size` then there is no replacement
        :return:
        """
        sample_size = 5
        population = list(range(10))
        population_generator = generator(population)
        samples = rnd.random_sample(population_generator, sample_size)

        self._default_tests(population, sample_size, samples)

        with self.subTest("Sampled without replacement"):
            self.assertEqual(len(set(samples)), len(samples))

    def test_sampled_with_replacement(self):
        """
        Check that if population is less than `sample_size` then there is replacement
        :return:
        """
        sample_size = 10
        population = list(range(5))
        population_generator = generator(population)
        samples = rnd.random_sample(population_generator, sample_size)

        self._default_tests(population, sample_size, samples)

        with self.subTest("sampled with replacement"):
            self.assertGreater(len(samples), len(set(samples)))

    def test_zero_population(self):
        sample_size = 10
        population = generator([])
        samples = rnd.random_sample(population, sample_size)
        self.assertEqual([], samples)
