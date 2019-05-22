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

import numpy as np


def get_context_values_to_put(context):

    context_values_to_put = {}

    for depth, node_list in context.items():

        values_to_put_at_this_depth = {}

        for node in node_list:

            values_to_put_for_this_node = {}

            if node.role_label is not None:
                values_to_put_for_this_node['role_type'] = node.role_label
            if node.role_direction is not None:
                values_to_put_for_this_node['role_direction'] = node.role_direction

            values_to_put_for_this_node['neighbour_type'] = node.thing.type_label

            if node.thing.data_type is not None:
                values_to_put_for_this_node['neighbour_data_type'] = node.thing.data_type
                values_to_put_for_this_node['neighbour_value_' + node.thing.data_type] = node.thing.value

            values_to_put_at_this_depth[node.indices] = values_to_put_for_this_node

        context_values_to_put[depth] = values_to_put_at_this_depth

    return context_values_to_put


def batch_values_to_put(batch_values):
    batched_values = {}
    for batch_index, structure in enumerate(batch_values):
        for depth, indexed_values_to_put in structure.items():
            for index, values_to_put in indexed_values_to_put.items():
                full_index = (batch_index,) + index
                batched_values.setdefault(depth, {})[full_index] = values_to_put

    return batched_values


def initialise_arrays(array_shape: typ.Tuple[int], **array_names_with_dtypes_and_default_values):

    if len(array_names_with_dtypes_and_default_values) == 0:
        raise ValueError('At least one array dtype and default value must be provided')

    arrays = {}
    for array_name, (array_data_type, default_value) in array_names_with_dtypes_and_default_values.items():

        arrays[array_name] = np.full(shape=array_shape,
                                     fill_value=default_value,
                                     dtype=array_data_type)
    return arrays


def initialise_arrays_for_all_depths(max_hops_shape: typ.Tuple[int], **array_names_with_dtypes_and_default_values):
    initialised_depth_arrays = []
    depth_array_sizes = get_depth_array_sizes(max_hops_shape)

    for i, array_shape in enumerate(depth_array_sizes):
        if i == len(depth_array_sizes) - 1:

            array_names_with_dtypes_and_default_values.pop('role_type', None)
            array_names_with_dtypes_and_default_values.pop('role_direction', None)

        initialised_depth_arrays.append(initialise_arrays(array_shape, **array_names_with_dtypes_and_default_values))

    return initialised_depth_arrays


def get_depth_array_sizes(max_hops_shape: typ.Tuple[int]):
    depth_array_sizes = []
    max_hops_size_list = list(max_hops_shape)
    for _ in max_hops_shape[1:]:

        depth_array_sizes.append(tuple(max_hops_size_list))

        max_hops_size_list.pop(1)
    return depth_array_sizes


def fill_arrays_at_all_depths(initialised_arrays, batch_values: typ.Dict):
    """
    Populates initialised arrays
    :param initialised_arrays: Arrays for the different hops of the context, for the different datatypes needed,
    initialised with default values
    :param batch_values: The sparse values to use to populate the arrays
    :return: Populated arrays
    """

    for depth, indexed_values_to_put in batch_values.items():
        for indices, values_to_put in indexed_values_to_put.items():
            for array_name, value_to_put in values_to_put.items():
                expanded_indices = indices + (0,)
                initialised_arrays[depth][array_name][expanded_indices] = value_to_put

    return initialised_arrays


def convert_context_batch_to_arrays(context_batch, max_hops_shape: typ.Tuple,
                                    **array_names_with_dtypes_and_default_values: typ.Tuple):
    indexed_values = map(get_context_values_to_put, context_batch)

    batch_values = batch_values_to_put(indexed_values)

    # Now we have a data structure like this:
    # {
    #     depth: {
    #         index: { values to put}
    #     }
    # }
    # Where the index now includes the number within the batch as its first element

    initialised_arrays = initialise_arrays_for_all_depths(max_hops_shape, **array_names_with_dtypes_and_default_values)

    return fill_arrays_at_all_depths(initialised_arrays, batch_values)
