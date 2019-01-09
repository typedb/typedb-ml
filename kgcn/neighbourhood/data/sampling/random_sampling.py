import random

"""
Brief:
From a given query, randomly or pseudo-randomly take N of the answers. If there are less than N results, then sample 
with replacement. 
-Ideally don't require asking for all answers to save time.
-We want the results to be streamed

"""

random.seed(1)


def random_sample(population, sample_size):
    """
    Samples items from an iterable population retrieving `sample_size` samples randomly without replacement. If
    `sample_size` samples aren't available only yield the first `sample_size` items
    :param population: An iterator of items to sample
    :param sample_size: The number of items to randomly retrieve from the population
    :return: A list of randomly selected items, possibly containing duplicates
    """
    results = []
    empty = True

    for i, item in enumerate(population):
        empty = False
        r = random.randint(0, i)
        if r < sample_size:
            if i < sample_size:
                results.insert(r, item)  # add first n items in random order
            else:
                results[r] = item  # at a decreasing rate, replace random items

    if empty:
        raise ValueError('Population is empty')

    return results
