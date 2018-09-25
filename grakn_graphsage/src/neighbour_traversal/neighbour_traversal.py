import itertools

import grakn


TARGET_PLAYS = 'target_plays'  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 'neighbour_plays'  # In this case the target

# Only needed due to a bug
UNKNOWN_ROLE_NEIGHBOUR_PLAYS = "UNKNOWN_ROLE_NEIGHBOUR_PLAYS"
UNKNOWN_ROLE_TARGET_PLAYS = "UNKNOWN_ROLE_TARGET_PLAYS"


class NeighbourRole:
    def __init__(self, role, neighbour, target_or_neighbour_plays):
        self.role = role
        self.neighbour = neighbour
        self.target_or_neighbour_plays = target_or_neighbour_plays


class ConceptWithNeighbourhood:
    def __init__(self, concept, neighbourhood):
        self.concept = concept
        self.neighbourhood = neighbourhood


def _get_neighbour_role(grakn_tx, role_and_concept_iterator, depth):
    for role, neighbour, target_or_neighbour_plays in role_and_concept_iterator:
        neighbour = build_neighbourhood_generator(grakn_tx, neighbour, depth - 1)
        yield NeighbourRole(role=role, neighbour=neighbour, target_or_neighbour_plays=target_or_neighbour_plays)


def build_neighbourhood_generator(grakn_tx: grakn.Transaction,
                                  target_concept: grakn.service.Session.Concept.Concept,
                                  depth: int):

    def _empty():
        yield from ()

    if depth == 0:
        # # This marks the end of the recursion, simply return this concept
        # return target_concept
        return ConceptWithNeighbourhood(concept=target_concept, neighbourhood=_empty())

    # Different cases for traversal

    # Any concept could play a role in a relationship if the schema permits it
    # TODO Inferred concepts have an id, but can we treat them exactly the same as non-inferred, or must we keep the
    # transaction open?

    # TODO Can't do this presently since querying for the role throws an exception
    # roles_played_iterator = grakn_tx.query("match $x id {}; $relationship($role: $x); get $relationship,
    # $role;".format(target_concept.id))
    roles_played_query = "match $x id {}; $relationship($x); get $relationship;".format(target_concept.id)
    print(roles_played_query)
    roles_played_iterator = grakn_tx.query(roles_played_query)

    def _roles_played_iterator():
        for answer in roles_played_iterator:

            # TODO See above, omitting due to bug
            # role_concept = answer.get("role")
            role_concept = UNKNOWN_ROLE_TARGET_PLAYS
            relationship_concept = answer.get("relationship")

            yield role_concept, relationship_concept, TARGET_PLAYS

    # Distinguish the concepts found as roles-played
    # Get them lazily

    concept_with_neighbourhood = ConceptWithNeighbourhood(concept=target_concept,
                                                          neighbourhood=_get_neighbour_role(grakn_tx,
                                                                                            _roles_played_iterator(),
                                                                                            depth))

    # TODO If user doesn't attach anything to impicit @has relationships, then these could be filtered out. Instead
    # another query would be required: "match $x id {}, has attribute $attribute; get $attribute;"
    # We would use TARGET_PLAYS with Role "has"

    # if node.is_entity():
    #     # Nothing special to do here?
    #     pass
    # if target_concept.is_attribute():
    #     # Do anything specific to attribute values
    #     # Optionally stop further propagation through attributes, since they are shared across the knowledge graph so
    #     # this may not provide relevant information
    #     neighbourhood.value = target_concept.value()

    if target_concept.is_relationship():
        # Find it's roleplayers
        # id and rel_type should be known (providing rel_type speeds up the query, but shouldn't since we provide the
        #  id)
        # Then from this list of roleplayers, remove `node`, since that's where we've come from
        # Distinguish the concepts found as roleplayers

        # TODO Can't do this presently since querying for the role throws an exception
        # roleplayers_query = "match $relationship id {}; $relationship($role: $x) isa {}; get $x, $role;".format(target_concept.id,
        #                                                                                         target_concept.type().label())
        roleplayers_query = "match $relationship id {}; $relationship($x) isa {}; get $x;".format(target_concept.id,
                                                                                                target_concept.type().label())
        print(roleplayers_query)
        roleplayers_iterator = grakn_tx.query(roleplayers_query)

        def _get_roleplayers_iterator():
            for answer in roleplayers_iterator:
                # role_concept = answer.get("role")
                role_concept = UNKNOWN_ROLE_NEIGHBOUR_PLAYS
                roleplayer_concept = answer.get("x")
                yield role_concept, roleplayer_concept, NEIGHBOUR_PLAYS

        # Chain the iterators together, so that after getting the roles played you get the roleplayers
        concept_with_neighbourhood.neighbourhood = itertools.chain(concept_with_neighbourhood.neighbourhood,
                                                                   _get_neighbour_role(grakn_tx,
                                                                                       _get_roleplayers_iterator(),
                                                                                       depth))
    return concept_with_neighbourhood


def collect_to_tree(concept_with_neighbourhood):
    """
    Given the neighbour generators, yield the fully populated tree of each of the target concept's neighbours
    :param neighbourhood_generators:
    :return:
    """
    if concept_with_neighbourhood is not None:
        neighbour_roles = set()
        for neighbour_role in concept_with_neighbourhood.neighbourhood:
            neighbour_roles.add(neighbour_role)
            collect_to_tree(neighbour_role.neighbour)

        concept_with_neighbourhood.neighbourhood = neighbour_roles

    return concept_with_neighbourhood


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
            m = get_max_depth(neighbour_role.neighbour)
            if m > max_depth:
                max_depth = m
        return max_depth + 1
