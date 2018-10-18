import itertools


def first_n_sample(population, sample_size):
    """
    Samples the population, taking the first `n` items (a.k.a `sample_size') encountered,
    :param population: An iterator of items to sample
    :param sample_size: The number of items to retrieve from the population
    :return: A list of the first `sample_size` items from the population
    """
    return list(itertools.islice(population, sample_size))