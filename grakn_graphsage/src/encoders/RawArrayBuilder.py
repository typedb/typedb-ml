from collections import OrderedDict

import numpy as np


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
            [('role_dir', np.int), ('role_type', np.int), ('thing_type', np.int), ('data_type', np.int)])

        self._value_data_types = OrderedDict(
            [('long', np.int), ('double', np.float), ('boolean', np.bool), ('date', np.datetime64),
             ('string', np.str)])  # TODO pass this in or make into class variable?

    def build_raw_arrays(self, concepts_with_neighbourhoods):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param concepts_with_neighbourhoods:
        :return: a list of arrays, one for each depth, including one for the starting nodes of interest
        """

        #####################################################
        # Make the empty arrays to fill
        #####################################################

        full_depth_shape = [self._n_starting_concepts] + self._neighbourhood_sizes

        shape_at_this_depth = []
        # for depth in range(self._max_depth):
        for dimension in full_depth_shape:
            shape_at_this_depth.append(dimension)
            matrices = {}
            for matrix_name, matrix_data_type in self._matrix_data_types.items():
                matrices[matrix_name] = np.full(shape_at_this_depth, fill_value=, dtype=matrix_data_type)

            # shape_at_this_depth = full_depth_shape[:depth + 1]
            # feats_at_this_depth = np.empty(shape_at_this_depth)

        #####################################################
        # Populate the arrays from the neighbour traversals
        #####################################################

        def build_neighbour_roles(neighbour_roles, depthwise_arrays, indices):

            depth = len(indices) - 1
            arrays_at_this_depth = depthwise_arrays[depth]

            for n, neighbour_role in enumerate(neighbour_roles):
                current_indices = tuple(indices + [n])  # Needs to be a tuple to index a numpy array

                thing_type_index = self._thing_type_labels.index(
                    neighbour_role.neighbour_with_neighbourhood.concept.type().label())

                arrays_at_this_depth.thing_type[current_indices] = thing_type_index

                role_type_index = self._role_type_labels.index(neighbour_role.role.label())

                role_direction = neighbour_role.role_direction

                value = RawArrayBuilder.DEFAULT_VALUE
                # ... etc for other data_types
                data_type_index = RawArrayBuilder.DEFAULT_DATATYPE_INDEX

                if neighbour_role.neighbour_with_neighbourhood.concept.is_attribute():
                    data_type_name = neighbour_role.neighbour_with_neighbourhood.concept.type().data_type()
                    data_type_index = list(self._value_data_types.keys()).index(data_type_name)

                    # Use data_type_index to index which matrix to write to
                    value_matrices[data_type_index]

                    depthwise_arrays = build_neighbour_roles(neighbour_role.neighbour_with_neighbourhood.neighbourhood,
                                                             depthwise_arrays, current_indices)

            return depthwise_arrays

        depthwise_arrays = []

        value_matrices = []
        for numpy_type in self._value_data_types.values():
            value_matrices.append(np.full(shape, RawArrayBuilder.DEFAULT_VALUE, dtype=numpy_type))

        for c, concept_with_neighbourhood in enumerate(concepts_with_neighbourhoods):
            # concept_tree = collect_to_tree(concept_with_neighbourhood)

            indices = [c]
            depthwise_arrays = build_neighbour_roles(concept_with_neighbourhood.neighbourhood, depthwise_arrays, indices)




            concept_with_neighbourhood.neighbourhood = [neighbour_role ]
            for neighbour_role in concept_with_neighbourhood.neighbourhood:
                collect_to_tree(neighbour_role.neighbour_with_neighbourhood)
