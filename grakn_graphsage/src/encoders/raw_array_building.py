import collections
import numpy as np
import grakn_graphsage.src.neighbourhood.traversal as trv


class RawArrayBuilder:

    DEFAULT_VALUE = np.nan
    DEFAULT_DATATYPE_INDEX = -1  # For use when a concept is not an attribute and hence doesn't have a data type

    def __init__(self, thing_type_labels, role_type_labels, neighbourhood_sizes, n_starting_concepts):
        """

        :param thing_type_labels: A list where the indices will be used to uniquely identify each thing label
        :param role_type_labels: A list where the indices will be used to uniquely identify each role label
        :param neighbourhood_sizes:
        :param n_starting_concepts:
        """
        self._thing_type_labels = thing_type_labels
        self._role_type_labels = role_type_labels
        self._neighbourhood_sizes = neighbourhood_sizes
        self._n_starting_concepts = n_starting_concepts

        self._matrix_data_types = collections.OrderedDict(
            [('role_direction', np.int), ('role_type', np.int), ('thing_type', np.int), ('data_type', np.int),
             ('value_long', np.int), ('value_double', np.float), ('value_boolean', np.bool),
             ('value_date', np.datetime64), ('value_string', np.str)])

    def build_raw_arrays(self, concepts_with_neighbourhoods):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param concepts_with_neighbourhoods:
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

        def _build_for_thing(thing, matrices_at_this_depth, current_indices):
            thing_type_index = self._thing_type_labels.index(
                thing.type().label())
            matrices_at_this_depth['thing_type'][current_indices] = thing_type_index

            data_type_index = RawArrayBuilder.DEFAULT_DATATYPE_INDEX

            if thing.is_attribute():
                data_type_name = thing.type().data_type().name.lower()

                # TODO Do any preprocessing of attribute values here ready to write them to numpy arrays
                data_type_index = list(self._matrix_data_types.keys()).index('value_' + data_type_name)

                matrices_at_this_depth['value_' + data_type_name][current_indices] = thing.value()

            # Store the index of the matrix the value was written to (which should match with its type)
            matrices_at_this_depth['data_type'][current_indices] = data_type_index

        def _build_neighbour_roles(neighbour_roles, depthwise_matrices, indices):

            depth = len(indices)

            for n, neighbour_role in enumerate(neighbour_roles):

                matrices_at_this_depth = depthwise_matrices[depth]
                current_indices = tuple(list(indices) + [n])  # Needs to be a tuple to index a numpy array

                if neighbour_role.role is not None:
                    role_type_index = self._role_type_labels.index(neighbour_role.role.label())
                    matrices_at_this_depth['role_type'][current_indices] = role_type_index

                if neighbour_role.role_direction is not None:
                    role_direction = neighbour_role.role_direction
                    matrices_at_this_depth['role_direction'][current_indices] = role_direction

                _build_for_thing(neighbour_role.neighbour_with_neighbourhood.concept, matrices_at_this_depth, current_indices)

                depthwise_matrices = _build_neighbour_roles(neighbour_role.neighbour_with_neighbourhood.neighbourhood,
                                                            depthwise_matrices, current_indices)
            return depthwise_matrices

        # Dummy NeighbourRoles so that a consistent data structure can be used right from the top level
        top_neighbour_roles = [trv.NeighbourRole(None, concept_with_neighbourhood, None) for concept_with_neighbourhood in
                               concepts_with_neighbourhoods]

        depthwise_matrices = _build_neighbour_roles(top_neighbour_roles, depthwise_matrices, [])
        return depthwise_matrices
