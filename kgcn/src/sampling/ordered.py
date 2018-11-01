
def ordered_sample(population, sample_size):
    """
    Samples the population, taking the first `n` items (a.k.a `sample_size') encountered. If more samples are
    requested than are available, repeat the first n until satisfied
    :param population: An iterator of items to sample
    :param sample_size: The number of items to retrieve from the population
    :return: A list of the first `sample_size` items from the population
    """

    results = []
    stored_items = []

    for i, item in enumerate(population):
        if i >= sample_size:
            break
        if i <= sample_size and stored_items is not None:
            # If we aren't yet sure if we have enough items then record the ones we've seen
            stored_items.append(item)
        else:
            stored_items = None

        results.append(item)  # add first n items in order

    if len(results) > 0:
        while len(results) < sample_size:
            # Now we start sampling with replacement
            n_additional_required = sample_size - len(results)
            # if n_additional_required >= len(results):
            #     results += results
            # else:
            results += results[:n_additional_required]
    else:
        return []

    return results
