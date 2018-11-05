import itertools

import collections

import kgcn.src.neighbourhood.data.executor
import kgcn.src.neighbourhood.data.executor as data_executor


class ConceptInfoWithNeighbourhood:
    def __init__(self, concept_info: kgcn.src.neighbourhood.data.executor.ConceptInfo, neighbourhood: collections.Generator):
        self.concept_info = concept_info
        self.neighbourhood = neighbourhood  # An iterator of `NeighbourRole`s


class NeighbourRole:
    def __init__(self, role_label: (str, None), role_direction: (int, None),
                 neighbour_info_with_neighbourhood: ConceptInfoWithNeighbourhood):
        self.role_label = role_label
        self.role_direction = role_direction
        self.neighbour_info_with_neighbourhood = neighbour_info_with_neighbourhood


def concepts_with_neighbourhoods_to_neighbour_roles(concept_infos_with_neighbourhoods):
    """Dummy NeighbourRoles so that a consistent data structure can be used right from the top level"""
    top_level_neighbour_roles = [NeighbourRole(None, None, concept_info_with_neighbourhood) for
                                 concept_info_with_neighbourhood in concept_infos_with_neighbourhoods]
    return top_level_neighbour_roles


class NeighbourhoodTraverser:
    def __init__(self, query_executor: data_executor.TraversalExecutor, depth_samplers):
        self._query_executor = query_executor
        self._depth_samplers = depth_samplers

    def __call__(self, target_concept_info: kgcn.src.neighbourhood.data.executor.ConceptInfo):
        depth = len(self._depth_samplers)
        return self._traverse(target_concept_info, depth)

    def _traverse(self, target_concept_info: kgcn.src.neighbourhood.data.executor.ConceptInfo, depth: int):

        def _empty():
            yield from ()

        if depth == 0:
            # This marks the end of the recursion, so there are no neighbours in the neighbourhood
            return ConceptInfoWithNeighbourhood(concept_info=target_concept_info, neighbourhood=_empty())

        sampler = self._depth_samplers[-depth]
        next_depth = depth - 1
        # next_neighbour_samplers = self._depth_samplers[1:]

        # Different cases for traversal

        # Any concept could play a role in a relationship if the schema permits it

        # Distinguish the concepts found as roles-played
        roles_played = self._query_executor(data_executor.ROLES_PLAYED, target_concept_info.id)

        neighbourhood = self._get_neighbour_role(roles_played, next_depth)

        concept_info_with_neighbourhood = ConceptInfoWithNeighbourhood(concept_info=target_concept_info,
                                                                       neighbourhood=neighbourhood)

        # TODO If user doesn't attach anything to impicit @has relationships, then these could be filtered out. Instead
        # another query would be required: "match $x id {}, has attribute $attribute; get $attribute;"
        # We would use TARGET_PLAYS with Role "has" or role "@has-< attribute type name >"

        # if target_concept_info.metatype_label == 'entity':
        #     # Nothing special to do in this case?
        #     pass
        # elif target_concept_info.metatype_label == 'attribute':
        #     # Do anything specific to attribute values
        #     # Optionally stop further propagation through attributes, since they are shared across the knowledge
        #     # graph so this may not provide relevant information

        if target_concept_info.base_type_label == 'relationship':
            # Find its roleplayers
            roleplayers = self._query_executor(data_executor.ROLEPLAYERS, target_concept_info.id, target_concept_info.type_label)

            # Chain the iterators together, so that after getting the roles played you get the roleplayers
            concept_info_with_neighbourhood.neighbourhood = itertools.chain(
                concept_info_with_neighbourhood.neighbourhood,
                self._get_neighbour_role(roleplayers, next_depth))

        # Randomly sample the neighbourhood
        concept_info_with_neighbourhood.neighbourhood = sampler(concept_info_with_neighbourhood.neighbourhood)

        return concept_info_with_neighbourhood

    def _get_neighbour_role(self, role_and_concept_info_iterator, depth):

        for connection in role_and_concept_info_iterator:
            neighbour_info_with_neighbourhood = self._traverse(connection['neighbour_info'], depth)

            yield NeighbourRole(role_label=connection['role_label'], role_direction=connection['role_direction'],
                                neighbour_info_with_neighbourhood=neighbour_info_with_neighbourhood)


def collect_to_tree(concept_info_with_neighbourhood):
    """
    Given the neighbour generators, yield the fully populated tree of each of the target concept's neighbours
    :param concept_info_with_neighbourhood:
    :return:
    """
    if concept_info_with_neighbourhood is not None:
        concept_info_with_neighbourhood.neighbourhood = materialise_subordinate_neighbours(
            concept_info_with_neighbourhood)
        for neighbour_role in concept_info_with_neighbourhood.neighbourhood:
            collect_to_tree(neighbour_role.neighbour_info_with_neighbourhood)

    return concept_info_with_neighbourhood


def materialise_subordinate_neighbours(concept_info_with_neighbourhood):
    """
    Build the list of all of the neighbours immediately "beneath" this concept. By beneath, meaning belonging to one
    layer deeper in the neighbour graph
    :param concept_info_with_neighbourhood:
    :return:
    """
    return [neighbour_role for neighbour_role in concept_info_with_neighbourhood.neighbourhood]


def get_max_depth(concept_info_with_neighbourhood):
    """
    Find the length of the deepest aggregation path
    :param concept_info_with_neighbourhood:
    :return:
    """

    if len(concept_info_with_neighbourhood.neighbourhood) == 0:
        return 0
    else:
        max_depth = 0
        for neighbour_role in concept_info_with_neighbourhood.neighbourhood:
            m = get_max_depth(neighbour_role.neighbour_info_with_neighbourhood)
            if m > max_depth:
                max_depth = m
        return max_depth + 1
