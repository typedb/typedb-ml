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

import copy
import typing as typ

import grakn
import tensorflow as tf

import kglib.kgcn.neighbourhood.data.executor as data_ex
import kglib.kgcn.neighbourhood.data.traversal as trv
import kglib.kgcn.preprocess.raw_array_builder as raw


def datetime_to_unixtime(datetime_array):
    return datetime_array.astype('datetime64[s]').astype('int64')


def apply_operations(raw_arrays, operations):
    """
    :param raw_arrays: numpy arrays to be transformed into a format such that can be fed into a TensorFlow op graph
    :return: tensor-ready arrays
    """

    preprocessed_raw_arrays = []
    for raw_array in raw_arrays:
        preprocessed_features = {}
        for key, features_array in raw_array.items():
            if key in set(operations.keys()):
                if operations[key] is not None:
                    preprocessed_features[key] = operations[key](features_array)
            else:
                preprocessed_features[key] = features_array

        preprocessed_raw_arrays.append(preprocessed_features)

    return preprocessed_raw_arrays


class KGCNFeature:
    def __init__(self, raw_data_type, formatter, tensorisor):
        self.raw_data_type = raw_data_type
        self.formatter = formatter
        self.tensorisor = tensorisor


F = KGCNFeature


class DatasetBuilder:
    def __init__(self, neighbour_sample_sizes,
                 role_type=F(tf.string, lambda x: x, lambda x: tf.convert_to_tensor(x, dtype=tf.string)),
                 role_direction=F(tf.int64, lambda x: x, lambda x: x),
                 neighbour_type=F(tf.string, lambda x: x, lambda x: tf.convert_to_tensor(x, dtype=tf.string)),
                 neighbour_data_type=F(tf.string, lambda x: x, lambda x: x),
                 neighbour_value_long=F(tf.int64, lambda x: x, lambda x: x),
                 neighbour_value_double=F(tf.float32, lambda x: x, lambda x: x),
                 neighbour_value_boolean=F(tf.int64, lambda x: x, lambda x: x),
                 neighbour_value_date=F(tf.int64, datetime_to_unixtime, lambda x: x),
                 neighbour_value_string=F(tf.string, lambda x: x, lambda x: x)):
        # TODO Get rid of pointless lambdas
        # TODO Remove formatters

        self._neighbour_sample_sizes = neighbour_sample_sizes

        self._features = {'role_type': role_type,
                          'role_direction': role_direction,
                          'neighbour_type': neighbour_type,
                          'neighbour_data_type': neighbour_data_type,
                          'neighbour_value_long': neighbour_value_long,
                          'neighbour_value_double': neighbour_value_double,
                          'neighbour_value_boolean': neighbour_value_boolean,
                          'neighbour_value_date': neighbour_value_date,
                          'neighbour_value_string': neighbour_value_string}

        for key, value in self._features.items():
            # Remove the features we don't want
            if value is None:
                del self._features[key]

        # self._formatters = {feature_name: feature.formatter for feature_name, feature in self._features.items()}
        self._feature_types = {feature_name: feature.raw_data_type for feature_name, feature in self._features.items()}
        self._tensorisors = {feature_name: feature.tensorisor for feature_name, feature in self._features.items()}

        ################################################################################################################
        # Placeholders
        ################################################################################################################

        all_feature_types = [copy.copy(self._feature_types) for _ in range(len(self._neighbour_sample_sizes) + 1)]
        # Remove role placeholders for the starting concepts (there are no roles for them)
        del all_feature_types[-1]['role_type']
        del all_feature_types[-1]['role_direction']

        # Build the placeholders for the neighbourhood_depths for each feature type
        self.raw_array_placeholders = build_array_placeholders(None, self._neighbour_sample_sizes, 1,
                                                               all_feature_types, name='array_input')

    def __call__(self):

        ################################################################################################################
        # Tensorising
        ################################################################################################################

        # Any steps needed to get arrays ready for the rest of the pipeline
        with tf.name_scope('tensorising') as scope:
            tensorised_arrays = apply_operations(self.raw_array_placeholders, self._tensorisors)

        ################################################################################################################
        # Build Dataset
        ################################################################################################################
        # Building the dataset using a generator became unnecessarily complex
        # https://stackoverflow.com/questions/51136862/creating-a-tensorflow-dataset-that-outputs-a-dict
        array_datasets = []
        for tensorised_array in tensorised_arrays:
            array_dataset = {}
            for key, tensor in tensorised_array.items():
                array_dataset[key] = tf.data.Dataset.from_tensor_slices(tensor)
            array_datasets.append(array_dataset)

        arrays_dataset = tf.data.Dataset.zip(tuple(array_datasets))

        return arrays_dataset, self.raw_array_placeholders


def build_feed_dict(raw_array_placeholders, raw_array_depths, labels_placeholder=None, labels=None):
    feed_dict = {}
    if labels is not None:
        feed_dict[labels_placeholder] = labels

    for raw_array_placeholder, raw_arrays_dict in zip(raw_array_placeholders, raw_array_depths):
        for feature_type_name in list(raw_arrays_dict.keys()):
            feed_dict[raw_array_placeholder[feature_type_name]] = raw_arrays_dict[feature_type_name]

    return feed_dict


def build_array_placeholders(batch_size, neighbourhood_sizes, features_length,
                             feature_types: typ.Union[typ.List[typ.MutableMapping[str, tf.DType]], tf.DType],
                             name=None):
    array_neighbourhood_sizes = list(reversed(neighbourhood_sizes))
    neighbourhood_placeholders = []

    histogram_allowed_dtypes = [tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64, tf.bfloat16,
                                tf.uint16, tf.float16, tf.uint32, tf.uint64]

    for i in range(len(array_neighbourhood_sizes) + 1):
        shape = [batch_size] + list(array_neighbourhood_sizes[i:]) + [features_length]
        phs = {}
        for name, data_type in feature_types[i].items():
            phs[name] = tf.placeholder(data_type, shape=shape, name=name)
            if data_type in histogram_allowed_dtypes:
                tf.summary.histogram('input/' + name, phs[name])

        neighbourhood_placeholders.append(phs)
    return neighbourhood_placeholders


class Traverser:

    def __init__(self, traversal_samplers):
        self._traversal_samplers = traversal_samplers
        neighbour_sample_sizes = tuple(sampler.sample_size for sampler in self._traversal_samplers)
        self._raw_builder = raw.RawArrayBuilder(neighbour_sample_sizes)

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
