#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import typing as typ

import collections

import grakn
import numpy as np

from kglib.kgcn.core.ingest.traverse.data import neighbourhood as trv, executor as data_ex


def build_default_arrays(neighbourhood_sizes, n_starting_concepts, array_data_types):
    depthwise_arrays = []
    depth_shape = list(neighbourhood_sizes) + [1]

    for i in range(len(depth_shape)):
        shape_at_this_depth = [n_starting_concepts] + depth_shape[i:]
        arrays = {}
        for array_name, (array_data_type, default_value) in array_data_types.items():

            if not (i == len(depth_shape) - 1 and array_name in ['role_direction', 'role_type']):
                # For the starting nodes we don't need to store roles
                arrays[array_name] = np.full(shape=shape_at_this_depth,
                                             fill_value=default_value,
                                             dtype=array_data_type)

        depthwise_arrays.append(arrays)
    return depthwise_arrays


def determine_values_to_put(role_label, role_direction, neighbour_type_label, neighbour_data_type,
                            neighbour_value):
    values_to_put = {}
    if role_label is not None:
        values_to_put['role_type'] = role_label
    if role_direction is not None:
        values_to_put['role_direction'] = role_direction

    values_to_put['neighbour_type'] = neighbour_type_label

    if neighbour_data_type is not None:
        # Potentially confusing to create an index of these arrays, since role type and direction will be omitted
        #  for the starting concepts
        # values_to_put['neighbour_data_type'] = list(self._array_data_types.keys()).index(
        #     'neighbour_value_' + neighbour_data_type)
        values_to_put['neighbour_data_type'] = neighbour_data_type
        values_to_put['neighbour_value_' + neighbour_data_type] = neighbour_value

    return values_to_put


class RawArrayBuilder:

    def __init__(self, neighbourhood_sizes: typ.Tuple[int]):
        """

        :param neighbourhood_sizes: The number of neighbours sampled at each recursion
        """
        self._neighbourhood_sizes = tuple(reversed(neighbourhood_sizes))

        # Array types and default values
        self._array_data_types = collections.OrderedDict(
            [('role_type', (np.dtype('U50'), '')),
             ('role_direction', (np.int, 0)),
             ('neighbour_type', (np.dtype('U50'), '')),
             ('neighbour_data_type', (np.dtype('U10'), '')),
             ('neighbour_value_long', (np.int, 0)),
             ('neighbour_value_double', (np.float, 0.0)),
             ('neighbour_value_boolean', (np.int, -1)),
             ('neighbour_value_date', ('datetime64[s]', '')),
             ('neighbour_value_string', (np.dtype('U50'), ''))])
        self.indices_visited = []

    def _initialise_arrays(self, n_starting_concepts):
        #####################################################
        # Make the empty arrays to fill
        #####################################################

        return build_default_arrays(self._neighbourhood_sizes, n_starting_concepts, self._array_data_types)

    def build_raw_arrays(self, concept_infos_with_neighbourhoods: typ.List[trv.NeighbourRole]):
        """
        Build the arrays to represent the depths of neighbour traversals.
        :param top_level_neighbour_roles:
        :return: a list of arrays, one for each depth, including one for the starting nodes of interest
        """

        n_starting_concepts = len(concept_infos_with_neighbourhoods)
        self.indices_visited = []
        depthwise_arrays = self._initialise_arrays(n_starting_concepts)

        #####################################################
        # Populate the arrays from the neighbour traversals
        #####################################################
        depthwise_arrays = self._build_neighbour_roles(concept_infos_with_neighbourhoods,
                                                       depthwise_arrays,
                                                       tuple())
        # try:
        #     poss = all_possible_indices(self._neighbourhood_sizes, n_starting_concepts)
        #     assert(set(self.indices_visited) == set(poss))
        # except AssertionError:
        #     raise AssertionError(
        #         f'\nPossible indices: \n{poss}\n=====\nVisited indices\n{self.indices_visited}\n=====\nMising '
        #         f'Indices\n{set(poss).difference(set(self.indices_visited))}')
        return depthwise_arrays

    def _build_neighbour_roles(self, neighbour_roles: typ.List[trv.NeighbourRole],
                               depthwise_arrays: typ.List[typ.Dict[str, np.ndarray]],
                               indices: typ.Tuple):

        # depth = len(self._neighbourhood_sizes) + 2 - (len(indices) + 1)

        n = None
        current_indices = None

        for n, neighbour_role in enumerate(neighbour_roles):
            if len(indices) == 0:
                current_indices = (n, 0)
            else:
                current_indices = list(indices)
                current_indices.insert(1, n)
                current_indices = tuple(current_indices)
            self.indices_visited.append(current_indices)
            depth = len(self._neighbourhood_sizes) + 2 - len(current_indices)
            arrays_at_this_depth = depthwise_arrays[depth]

            concept_info = neighbour_role.neighbour_info_with_neighbourhood.concept_info
            values_to_put = determine_values_to_put(neighbour_role.role_label, neighbour_role.role_direction,
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

        print(f'n = {n}, indices = {current_indices}')

        # Duplicate the sections of the arrays already built so that they are padded to be complete
        if n is not None and depth < len(self._neighbourhood_sizes):
            expected_n = self._neighbourhood_sizes[depth] - 1
            if n < expected_n:
                boundary = n + 1
                slice_to_repeat = list(current_indices)
                slice_to_repeat[1] = slice(boundary)
                slice_to_repeat.insert(1, ...)
                slice_to_repeat = tuple(slice_to_repeat)

                slice_to_replace = list(slice_to_repeat)
                slice_to_replace[2] = slice(boundary, None)
                slice_to_replace = tuple(slice_to_replace)

                # For the current depth and deeper
                for d in list(range(depth, -1, -1)):
                    for array in list(depthwise_arrays[d].values()):
                        fill_array_with_repeats(array, slice_to_repeat, slice_to_replace)

        return depthwise_arrays


def fill_array_with_repeats(array, slice_to_repeat, slice_to_replace):
    to_repeat = array[slice_to_repeat]
    to_fill = array[slice_to_replace]

    num_repeats = -(-to_fill.shape[0] // to_repeat.shape[0])

    tile_axes = [1] * len(to_fill.shape)
    tile_axes[0] = num_repeats + 1

    filler = np.tile(to_repeat, tile_axes)

    filler_axes = tuple(slice(None, i) for i in to_fill.shape)

    curtailed_filler = filler[filler_axes]

    array[slice_to_replace] = curtailed_filler


class Traverser:

    def __init__(self, traversal_samplers):
        self._traversal_samplers = traversal_samplers
        neighbour_sample_sizes = tuple(sampler.sample_size for sampler in self._traversal_samplers)
        self._raw_builder = RawArrayBuilder(neighbour_sample_sizes)

    def __call__(self, session, concepts):
        ################################################################################################################
        # Neighbour Traversals
        ################################################################################################################
        concept_infos = [data_ex.build_concept_info(concept) for concept in concepts]

        data_executor = data_ex.TraversalExecutor()
        neighourhood_traverser = trv.NeighbourhoodTraverser(data_executor, self._traversal_samplers)

        neighbourhood_depths = []
        for concept_info in concept_infos:
            tx = session.transaction(grakn.TxType.WRITE)
            print(f'Opening transaction {tx}')
            depths = neighourhood_traverser(concept_info, tx)
            trv.collect_to_tree(depths)
            neighbourhood_depths.append(depths)
            print(f'closing transaction {tx}')
            tx.close()
        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################
        # TODO Deal with where to build arrays

        raw_array_depths = self._raw_builder.build_raw_arrays(neighbour_roles)
        return raw_array_depths
