
def ordered_sample(population, sample_size):
    """
    Samples the population, taking the first `n` items (a.k.a `sample_size') encountered. If more samples are
    requested than are available then only yield the first `sample_size` items
    :param population: An iterator of items to sample
    :param sample_size: The number of items to retrieve from the population
    :return: A list of the first `sample_size` items from the population
    """

    empty = True

    for i, item in enumerate(population):
        empty = False
        if i >= sample_size:
            break
        yield item

    if empty:
        raise ValueError('Population is empty')
