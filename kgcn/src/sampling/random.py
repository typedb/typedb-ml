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
    Samples items from an iterable population retrieving `sample_size` samples randomly without replacement. If `sample_size`
    samples aren't available then the items are sampled with replacement until `sample_size` samples have been
    selection (in this case there will therefore be repetition)
    :param population: An iterator of items to sample
    :param sample_size: The number of items to randomly retrieve from the population
    :return: A list of randomly selected items, possibly containing duplicates
    """
    results = []
    stored_items = []

    for i, item in enumerate(population):
        if i <= sample_size and stored_items is not None:
            # If we aren't yet sure if we have enough items then record the ones we've seen
            stored_items.append(item)
        else:
            stored_items = None

        r = random.randint(0, i)
        if r < sample_size:
            if i < sample_size:
                results.insert(r, item)  # add first n items in random order
            else:
                results[r] = item  # at a decreasing rate, replace random items
    if len(results) > 0:
        while len(results) < sample_size:
            # Now we start sampling with replacement
            n_additional_required = sample_size - len(results)
            # TODO calling random_sample recursively looks memory inefficient
            results += random_sample(stored_items, n_additional_required)
    else:
        return []

    return results
