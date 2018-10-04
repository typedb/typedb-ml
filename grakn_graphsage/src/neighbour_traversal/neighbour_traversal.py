import itertools

import grakn

from grakn_graphsage.src.random_sampling.random_sampling import random_sample

TARGET_PLAYS = 'target_plays'  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 'neighbour_plays'  # In this case the target

# Only needed due to a bug
UNKNOWN_ROLE_NEIGHBOUR_PLAYS = "UNKNOWN_ROLE_NEIGHBOUR_PLAYS"
UNKNOWN_ROLE_TARGET_PLAYS = "UNKNOWN_ROLE_TARGET_PLAYS"

MAX_LIMIT = 10000


class NeighbourRole:
    def __init__(self, role, neighbour_with_neighbourhood, role_direction):
        self.role = role
        self.neighbour_with_neighbourhood = neighbour_with_neighbourhood
        self.role_direction = role_direction


class ConceptWithNeighbourhood:
    def __init__(self, concept, neighbourhood):
        self.concept = concept
        self.neighbourhood = neighbourhood  # An iterator of `NeighbourRole`s


def get_roles_played(grakn_tx, target_concept, limit):

    # TODO Can't do this presently since querying for the role throws an exception
    # roles_played_query = (
    #     f"match $x id {target_concept.id}; $relationship($role: $x); offset {0}; limit {limit}; get $relationship, "
    #     f"$role;")

    roles_played_query = (
        f"match $x id {target_concept.id}; $relationship($x); offset {0}; limit {limit}; get $relationship;")

    print(roles_played_query)
    roles_played_iterator = grakn_tx.query(roles_played_query)

    def _roles_played_iterator():
        for answer in roles_played_iterator:

            # TODO See above, omitting due to bug
            # role_concept = answer.get("role")
            role_concept = UNKNOWN_ROLE_TARGET_PLAYS
            relationship_concept = answer.get("relationship")

            yield role_concept, relationship_concept, TARGET_PLAYS

    return _roles_played_iterator()


def get_roleplayers(grakn_tx, target_concept, limit):
    # id and the concept type should be known (providing the concept type speeds up the query, which it shouldn't
    # since we provide the concept's id)

    # TODO Can't do this presently since querying for the role throws an exception
    # roleplayers_query = (
    #     f"match $relationship id {target_concept.id}; $relationship($role: $x) isa {target_concept.type().label()};
    # offset {0}; limit {limit}; get $x, $role;")
    
    roleplayers_query = (
        f"match $relationship id {target_concept.id}; $relationship($x) isa {target_concept.type().label()}; "
        f"offset {0}; limit {limit}; get $x;")
    print(roleplayers_query)
    roleplayers_iterator = grakn_tx.query(roleplayers_query)

    def _get_roleplayers_iterator():
        for answer in roleplayers_iterator:
            # role_concept = answer.get("role")
            role_concept = UNKNOWN_ROLE_NEIGHBOUR_PLAYS
            roleplayer_concept = answer.get("x")
            yield role_concept, roleplayer_concept, NEIGHBOUR_PLAYS

    return _get_roleplayers_iterator()


def _get_neighbour_role(grakn_tx, role_and_concept_iterator, neighbour_sample_sizes, **kwargs):
    for role, neighbour, role_direction in role_and_concept_iterator:
        neighbour_with_neighbourhood = build_neighbourhood_generator(grakn_tx, neighbour, neighbour_sample_sizes,
                                                                     **kwargs)
        yield NeighbourRole(role=role, neighbour_with_neighbourhood=neighbour_with_neighbourhood,
                            role_direction=role_direction)


def build_neighbourhood_generator(grakn_tx: grakn.Transaction,
                                  target_concept: grakn.service.Session.Concept.Concept,
                                  neighbour_sample_sizes: tuple, limit_factor=2):

    def _empty():
        yield from ()

    depth = len(neighbour_sample_sizes)

    if depth == 0:
        # # This marks the end of the recursion, simply return this concept
        # return target_concept
        return ConceptWithNeighbourhood(concept=target_concept, neighbourhood=_empty())

    sample_size = neighbour_sample_sizes[0]
    next_neighbour_sample_sizes = neighbour_sample_sizes[1:]

    # Rather than looking at all roleplayers and roles played, limit the number to a multiple of the number of samples
    # wanted. This makes the process pseudo-random, but saves a lot of time when querying
    if limit_factor is None:
        limit = MAX_LIMIT
    else:
        limit = sample_size * limit_factor

    # Different cases for traversal

    # Any concept could play a role in a relationship if the schema permits it
    # TODO Inferred concepts have an id, but can we treat them exactly the same as non-inferred, or must we keep the
    # transaction open?

    # Distinguish the concepts found as roles-played
    # Get them lazily
    roles_played = get_roles_played(grakn_tx, target_concept, limit)

    neighbourhood = _get_neighbour_role(grakn_tx, roles_played, next_neighbour_sample_sizes, limit_factor=limit_factor)
    concept_with_neighbourhood = ConceptWithNeighbourhood(concept=target_concept, neighbourhood=neighbourhood)

    # TODO If user doesn't attach anything to impicit @has relationships, then these could be filtered out. Instead
    # another query would be required: "match $x id {}, has attribute $attribute; get $attribute;"
    # We would use TARGET_PLAYS with Role "has" or role "@has-< attribute type name >"

    # if node.is_entity():
    #     # Nothing special to do in this case?
    #     pass
    # if target_concept.is_attribute():
    #     # Do anything specific to attribute values
    #     # Optionally stop further propagation through attributes, since they are shared across the knowledge graph so
    #     # this may not provide relevant information

    if target_concept.is_relationship():
        # Find it's roleplayers
        roleplayers = get_roleplayers(grakn_tx, target_concept, limit)

        # Chain the iterators together, so that after getting the roles played you get the roleplayers
        concept_with_neighbourhood.neighbourhood = itertools.chain(concept_with_neighbourhood.neighbourhood,
                                                                   _get_neighbour_role(grakn_tx,
                                                                                       roleplayers,
                                                                                       next_neighbour_sample_sizes))

    # Randomly sample the neighbourhood
    concept_with_neighbourhood.neighbourhood = random_sample_generator(concept_with_neighbourhood.neighbourhood,
                                                                       sample_size)
    return concept_with_neighbourhood


def random_sample_generator(population, sample_size):
    """
    Just a wrapper for `random_sample` to make a generator
    """
    samples = random_sample(population, sample_size)
    for sample in samples:
        yield sample


def collect_to_tree(concept_with_neighbourhood):
    """
    Given the neighbour generators, yield the fully populated tree of each of the target concept's neighbours
    :param concept_with_neighbourhood:
    :return:
    """
    if concept_with_neighbourhood is not None:
        concept_with_neighbourhood.neighbourhood = materialise_subordinate_neighbours(concept_with_neighbourhood)
        for neighbour_role in concept_with_neighbourhood.neighbourhood:
            collect_to_tree(neighbour_role.neighbour_with_neighbourhood)

    return concept_with_neighbourhood


def materialise_subordinate_neighbours(concept_with_neighbourhood):
    """
    Build the list of all of the neighbours immediately "beneath" this concept. By beneath, meaning belonging to one
    layer deeper in the neighbour graph
    :param concept_with_neighbourhood:
    :return:
    """
    return [neighbour_role for neighbour_role in concept_with_neighbourhood.neighbourhood]


def get_max_depth(concept_with_neighbourhood):
    """
    Find the length of the deepest aggregation path
    :param concept_with_neighbourhood:
    :return:
    """

    if len(concept_with_neighbourhood.neighbourhood) == 0:
        return 0
    else:
        max_depth = 0
        for neighbour_role in concept_with_neighbourhood.neighbourhood:
            m = get_max_depth(neighbour_role.neighbour_with_neighbourhood)
            if m > max_depth:
                max_depth = m
        return max_depth + 1
