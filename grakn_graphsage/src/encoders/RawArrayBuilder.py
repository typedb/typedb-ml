from collections import OrderedDict

import numpy as np

from grakn_graphsage.src.neighbour_traversal.neighbour_traversal import NeighbourRole


class ArraysAtDepth:
    def __init__(self, role_type, role_dir, thing_type, data_type, values, long_value, double_value, boolean_value, date_value,
                 string_value):
        self.role_dir = role_dir
        self.role_type = role_type
        self.thing_type = thing_type

        self.data_type = data_type

        # # data_type = 0 for non-attributes, TODO or use None/NaN and then zero-index instead?
        # self.long_value = long_value  # data_type = 1
        # self.double_value = double_value  # data_type = 2
        # self.boolean_value = boolean_value  # data_type = 3
        # self.date_value = date_value  # data_type = 4
        # self.string_value = string_value  # data_type = 5

        self._values = values


class RawArrayBuilder:

    DEFAULT_VALUE = np.nan
    DEFAULT_DATATYPE_INDEX = np.nan

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
        self._max_depth = len(neighbourhood_sizes) + 1

        self._matrix_data_types = OrderedDict(
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
        full_depth_shape = [self._n_starting_concepts] + self._neighbourhood_sizes

        shape_at_this_depth = []
        # for depth in range(self._max_depth):
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

        def build_for_thing(thing, matrices_at_this_depth, current_indices):
            thing_type_index = self._thing_type_labels.index(
                thing.type().label())
            matrices_at_this_depth['thing_type'][current_indices] = thing_type_index

            data_type_index = RawArrayBuilder.DEFAULT_DATATYPE_INDEX

            if thing.is_attribute():
                data_type_name = thing.type().data_type()

                # TODO Do any preprocessing of attribute values here ready to write them to numpy arrays
                data_type_index = list(self._matrix_data_types.keys()).index('value_' + data_type_name)

                matrices_at_this_depth['value_' + data_type_name][
                    current_indices] = thing.value()

            # Store the index of the matrix the value was written to (which should match with its type)
            matrices_at_this_depth['data_type'][current_indices] = data_type_index

        def build_neighbour_roles(neighbour_roles, depthwise_matrices, indices):

            depth = len(indices)
            matrices_at_this_depth = depthwise_matrices[depth]

            for n, neighbour_role in enumerate(neighbour_roles):
                current_indices = tuple(indices + [n])  # Needs to be a tuple to index a numpy array

                if depth > 0:
                    role_type_index = self._role_type_labels.index(neighbour_role.role.label())
                    matrices_at_this_depth['role_type'][current_indices] = role_type_index

                    role_direction = neighbour_role.role_direction
                    matrices_at_this_depth['role_direction'][current_indices] = role_direction

                    concept_with_neighbourhood = neighbour_role
                else:
                    concept_with_neighbourhood = neighbour_role.neighbour_with_neighbourhood

                build_for_thing(concept_with_neighbourhood, matrices_at_this_depth, current_indices)

                depthwise_matrices = build_neighbour_roles(concept_with_neighbourhood.neighbourhood,
                                                           depthwise_matrices, current_indices)

            return depthwise_matrices


        for c, concept_with_neighbourhood in enumerate(concepts_with_neighbourhoods):
            # concept_tree = collect_to_tree(concept_with_neighbourhood)

            # TODO Should I change the top level concepts into the form of NeighbourRoles, or treat the top level as a special case?

            build_for_thing(concept_with_neighbourhood.concept, [], depthwise_matrices[0])
            indices = [c]
            depthwise_matrices = build_neighbour_roles(concept_with_neighbourhood.neighbourhood, depthwise_matrices, indices)



            concept_with_neighbourhood.neighbourhood = [neighbour_role ]
            for neighbour_role in concept_with_neighbourhood.neighbourhood:
                collect_to_tree(neighbour_role.neighbour_with_neighbourhood)
