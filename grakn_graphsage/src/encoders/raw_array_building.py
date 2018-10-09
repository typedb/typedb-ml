import collections
import numpy as np
import typing as typ
import grakn_graphsage.src.neighbourhood.traversal as trv


class RawArrayBuilder:

    DEFAULT_VALUE = -5
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
            [('role_type', np.int), ('role_direction', np.int), ('neighbour_type', np.int), ('neighbour_data_type', np.int),
             ('neighbour_value_long', np.int), ('neighbour_value_double', np.float), ('neighbour_value_boolean', np.bool),
             ('neighbour_value_date', np.datetime64), ('neighbour_value_string', np.dtype('U25'))])

    def _initialise_arrays(self):
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
        return depthwise_matrices

    def build_raw_arrays(self, top_level_neighbour_roles: typ.List[trv.NeighbourRole]):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param top_level_neighbour_roles:
        :return: a list of arrays, one for each depth, including one for the starting nodes of interest
        """

        depthwise_matrices = self._initialise_arrays()

        #####################################################
        # Populate the arrays from the neighbour traversals
        #####################################################

        depthwise_matrices = self._build_neighbour_roles(top_level_neighbour_roles, depthwise_matrices, tuple())
        return depthwise_matrices

    def _determine_values_to_put(self, role_label, role_direction, neighbour_type_label, neighbour_data_type,
                                 neighbour_value):
        values_to_put = {}
        if role_label is not None:
            values_to_put['role_type'] = self._role_type_labels.index(role_label)
        if role_direction is not None:
            values_to_put['role_direction'] = role_direction

        values_to_put['neighbour_type'] = self._thing_type_labels.index(neighbour_type_label)

        if neighbour_data_type is not None:
            values_to_put['neighbour_data_type'] = list(self._matrix_data_types.keys()).index('neighbour_value_' + neighbour_data_type)
            values_to_put['neighbour_value_' + neighbour_data_type] = neighbour_value

        return values_to_put

    # def _put_in_matrices(self, matrices, values_to_put, indices):
    #     for key, value in values_to_put.items():
    #         matrices[key][indices] = value
    #     return matrices

    # def _put_in_matrix(self, matrix, indices, value):
    #     matrix[indices] = value
    #     return matrix

    def _build_neighbour_roles(self, neighbour_roles: typ.List[trv.NeighbourRole],
                               depthwise_matrices: typ.List[typ.Dict[str, np.ndarray]],
                               indices: typ.Tuple):
        depth = len(indices)

        for n, neighbour_role in enumerate(neighbour_roles):

            matrices_at_this_depth = depthwise_matrices[depth]
            current_indices = tuple(list(indices) + [n])  # Needs to be a tuple to index a numpy array

            concept_info = neighbour_role.neighbour_info_with_neighbourhood.concept_info
            values_to_put = self._determine_values_to_put(neighbour_role.role_label, neighbour_role.role_direction,
                                                          concept_info.type_label, concept_info.data_type,
                                                          concept_info.value)

            for key, value in values_to_put.items():
                matrices_at_this_depth[key][current_indices] = value

            depthwise_matrices = self._build_neighbour_roles(
                neighbour_role.neighbour_info_with_neighbourhood.neighbourhood,
                depthwise_matrices, current_indices)

        return depthwise_matrices
