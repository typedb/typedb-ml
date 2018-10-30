
class DataTraversalStrategy:

    TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
    NEIGHBOUR_PLAYS = 1  # In this case the target

    # TODO Changing queries due to bug
    # query = "match $x id {}; $relationship($role: $x); offset {}; limit {}; get $relationship, $role;")
    ROLES_PLAYED_QUERY = {
        'query': "match $x id {}; $relationship($x); offset {}; limit {}; get $relationship;",
        'role_variable': 'role',
        'role_direction': TARGET_PLAYS,
        'neighbour_variable': 'relationship'}

    # query = "match $relationship id {}; $relationship($role: $x) isa {}; offset {}; limit {}; get $x, $role;"
    ROLEPLAYERS_QUERY = {
        'query': "match $relationship id {}; $relationship($x) isa {}; offset {}; limit {}; get $x;",
        'role_variable': 'role',
        'role_direction': NEIGHBOUR_PLAYS,
        'neighbour_variable': 'x'}

    def __init__(self, neighbour_sample_sizes, sampler, limit_factor=None, max_samples_limit=10000,
                 roles_played_query=ROLES_PLAYED_QUERY,
                 roleplayers_query=ROLEPLAYERS_QUERY):
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

        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query


