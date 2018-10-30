
class DataTraversalStrategy:

    ROLES_PLAYED_QUERY = (
            "match $x id {}; $relationship($x); offset {}; limit {}; get $relationship;")

    def __init__(self, neighbour_sample_sizes, sampler, limit_factor=None, max_samples_limit=10000):
        """
        Strategy to determine how the knowledge graph is traversed. Used to store parameters
        :param neighbour_sample_sizes: the number of neighbours to sample at each depth
        :param sampler: method of sampling neighbours
        :param max_samples_limit: The number of neighbours to randomly sample from. Reducing can increase traversal
        sampling speed but sacrifices randomness
        """
        self.sampler = sampler
        self.neighbour_sample_sizes = neighbour_sample_sizes
        self.limit_factor = limit_factor
        self.max_samples_limit = max_samples_limit
