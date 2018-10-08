import collections
import numpy as np
import typing as typ
import grakn_graphsage.src.neighbourhood.traversal as trv
import grakn_graphsage.src.neighbourhood.concept as concept


class RawArrayBuilder:

    DEFAULT_VALUE = np.nan
    DEFAULT_DATATYPE_INDEX = -1  # For use when a concept is not an attribute and hence doesn't have a data type

    def __init__(self,
                 thing_type_labels: typ.List[str],
                 role_type_labels: typ.List[str],
                 neighbourhood_sizes: typ.Tuple[int],
                 n_starting_concepts: int):
        """

        :param thing_type_labels: A list the index of which will be used to uniquely identify each thing label
        :param role_type_labels: A list the index of which will be used to uniquely identify each role label
        :param neighbourhood_sizes: The number of neighbours sampled at each recursion
        :param n_starting_concepts: number of concepts whose traversals are supplied
        """
        self._thing_type_labels = thing_type_labels
        self._role_type_labels = role_type_labels
        self._neighbourhood_sizes = neighbourhood_sizes
        self._n_starting_concepts = n_starting_concepts

        self._matrix_data_types = collections.OrderedDict(
            [('role_direction', np.int), ('role_type', np.int), ('thing_type', np.int), ('data_type', np.int),
             ('value_long', np.int), ('value_double', np.float), ('value_boolean', np.bool),
             ('value_date', np.datetime64), ('value_string', np.str)])

    def build_raw_arrays(self, top_level_neighbour_roles: typ.List[trv.NeighbourRole]):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param top_level_neighbour_roles:
        :return: a list of arrays, one for each depth, including one for the starting nodes of interest
        """

        #####################################################
        # Make the empty arrays to fill
        #####################################################

        depthwise_matrices = []
        full_depth_shape = [self._n_starting_concepts] + list(self._neighbourhood_sizes)

        shape_at_this_depth = []
        for dimension in full_depth_shape:
            shape_at_this_depth.append(dimension)
            matrices = {}
            for matrix_name, matrix_data_type in self._matrix_data_types.items():

                if dimension == 0 and matrix_name in ['role_direction', 'role_type']:
                    # For the starting nodes we don't need to store roles
                    matrices[matrix_name] = None
                else:
                    matrices[matrix_name] = np.full(shape=shape_at_this_depth,
                                                    fill_value=RawArrayBuilder.DEFAULT_VALUE,
                                                    dtype=matrix_data_type)

            depthwise_matrices.append(matrices)

        #####################################################
        # Populate the arrays from the neighbour traversals
        #####################################################

        depthwise_matrices = self._build_neighbour_roles(top_level_neighbour_roles, depthwise_matrices, tuple())
        return depthwise_matrices

    def _build_for_thing(self, thing_info: concept.ConceptInfo, matrices_at_this_depth: typ.Dict[str, np.ndarray],
                             current_indices: typ.Tuple[int]):

        thing_type_index = self._thing_type_labels.index(
            thing_info.type_label)
        matrices_at_this_depth['thing_type'][current_indices] = thing_type_index

        data_type_index = RawArrayBuilder.DEFAULT_DATATYPE_INDEX

        if thing_info.base_type_label == 'attribute':

            # TODO Do any preprocessing of attribute values here ready to write them to numpy arrays
            data_type_index = list(self._matrix_data_types.keys()).index('value_' + thing_info.data_type)

            matrices_at_this_depth['value_' + thing_info.data_type][current_indices] = thing_info.value

        # Store the index of the matrix the value was written to (which should match with its type)
        matrices_at_this_depth['data_type'][current_indices] = data_type_index
        return matrices_at_this_depth

    def _build_neighbour_roles(self, neighbour_roles: typ.List[trv.NeighbourRole],
                               depthwise_matrices: typ.List[typ.Dict[str, np.ndarray]],
                               indices: typ.Tuple):
        depth = len(indices)

        for n, neighbour_role in enumerate(neighbour_roles):

            matrices_at_this_depth = depthwise_matrices[depth]
            current_indices = tuple(list(indices) + [n])  # Needs to be a tuple to index a numpy array

            if neighbour_role.role_label is not None:
                role_type_index = self._role_type_labels.index(neighbour_role.role_label)
                matrices_at_this_depth['role_type'][current_indices] = role_type_index

            if neighbour_role.role_direction is not None:
                role_direction = neighbour_role.role_direction
                matrices_at_this_depth['role_direction'][current_indices] = role_direction

            matrices_at_this_depth = self._build_for_thing(
                neighbour_role.neighbour_info_with_neighbourhood.concept_info,
                matrices_at_this_depth,
                current_indices)

            depthwise_matrices = self._build_neighbour_roles(
                neighbour_role.neighbour_info_with_neighbourhood.neighbourhood,
                depthwise_matrices, current_indices)
        return depthwise_matrices
