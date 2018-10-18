import collections
import numpy as np
import typing as typ
import kgcn.src.neighbourhood.data.traversal as trv


class RawArrayBuilder:

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

        # Array types and default values
        self._array_data_types = collections.OrderedDict(
            [('role_type', (np.int, 0)), ('role_direction', (np.int, 0)), ('neighbour_type', (np.int, 0)),
             ('neighbour_data_type', (np.int, -1)), ('neighbour_value_long', (np.int, 0)),
             ('neighbour_value_double', (np.float, 0.0)), ('neighbour_value_boolean', (np.int, -1)),
             ('neighbour_value_date', (np.datetime64, '')), ('neighbour_value_string', (np.dtype('U25'), ''))])

    def _initialise_arrays(self):
        #####################################################
        # Make the empty arrays to fill
        #####################################################

        depthwise_arrays = []
        depth_shape = list(self._neighbourhood_sizes) + [1]

        for i in range(len(depth_shape)):
            shape_at_this_depth = [self._n_starting_concepts] + depth_shape[i:]
            arrays = {}
            for array_name, (array_data_type, default_value) in self._array_data_types.items():

                if i == len(depth_shape) - 1 and array_name in ['role_direction', 'role_type']:
                    # For the starting nodes we don't need to store roles
                    arrays[array_name] = None
                else:
                    arrays[array_name] = np.full(shape=shape_at_this_depth,
                                                 fill_value=default_value,
                                                 dtype=array_data_type)

            depthwise_arrays.append(arrays)
        return depthwise_arrays

    def build_raw_arrays(self, concept_infos_with_neighbourhoods: typ.List[trv.NeighbourRole]):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param top_level_neighbour_roles:
        :return: a list of arrays, one for each depth, including one for the starting nodes of interest
        """

        depthwise_arrays = self._initialise_arrays()

        #####################################################
        # Populate the arrays from the neighbour traversals
        #####################################################
        depthwise_arrays = self._build_neighbour_roles(concept_infos_with_neighbourhoods,
                                                       depthwise_arrays,
                                                       tuple())
        return depthwise_arrays

    def _build_neighbour_roles(self, neighbour_roles: typ.List[trv.NeighbourRole],
                               depthwise_arrays: typ.List[typ.Dict[str, np.ndarray]],
                               indices: typ.Tuple):

        for n, neighbour_role in enumerate(neighbour_roles):
            if len(indices) == 0:
                current_indices = (n, 0)
            else:
                current_indices = tuple([indices[0], n] + list(indices[1:]))

            depth = len(self._neighbourhood_sizes) + 2 - len(current_indices)
            arrays_at_this_depth = depthwise_arrays[depth]

            concept_info = neighbour_role.neighbour_info_with_neighbourhood.concept_info
            values_to_put = self._determine_values_to_put(neighbour_role.role_label, neighbour_role.role_direction,
                                                          concept_info.type_label, concept_info.data_type,
                                                          concept_info.value)

            for key, value in values_to_put.items():
                # Ensure that the rank of the array is the same as the number of indices, or risk setting more than
                # one value
                assert len(arrays_at_this_depth[key].shape) == len(current_indices)
                arrays_at_this_depth[key][current_indices] = value

            depthwise_arrays = self._build_neighbour_roles(
                neighbour_role.neighbour_info_with_neighbourhood.neighbourhood,
                depthwise_arrays,
                current_indices)

        return depthwise_arrays

    def _determine_values_to_put(self, role_label, role_direction, neighbour_type_label, neighbour_data_type,
                                 neighbour_value):
        values_to_put = {}
        if role_label is not None:
            values_to_put['role_type'] = self._role_type_labels.index(role_label)
        if role_direction is not None:
            values_to_put['role_direction'] = role_direction

        values_to_put['neighbour_type'] = self._thing_type_labels.index(neighbour_type_label)

        if neighbour_data_type is not None:
            values_to_put['neighbour_data_type'] = list(self._array_data_types.keys()).index(
                'neighbour_value_' + neighbour_data_type)
            values_to_put['neighbour_value_' + neighbour_data_type] = neighbour_value

        return values_to_put
