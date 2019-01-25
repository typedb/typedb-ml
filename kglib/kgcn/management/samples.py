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

import collections

import numpy as np

from kglib.kgcn.neighbourhood.data.sampling import random_sampling as random
from kglib.kgcn.use_cases.attribute_prediction import label_extraction as label_extraction


def query_for_random_samples_with_attribute(tx, query, example_var_name, attribute_var_name, attribute_vals,
                                            sample_size_per_label, population_size):
    concepts = {}
    labels = {}

    for a in attribute_vals:
        target_concept_query = query.format(a)

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           (example_var_name, collections.OrderedDict(
                                                               [(attribute_var_name, attribute_vals)])),
                                                           sampling_method=random.random_sample)
        concepts_with_labels = extractor(tx, sample_size_per_label, population_size)
        if len(concepts_with_labels) == 0:
            raise RuntimeError(f'Couldn\'t find any concepts to match target query "{target_concept_query}"')

        concepts[a] = [concepts_with_label[0] for concepts_with_label in concepts_with_labels]
        # TODO Should this be an array not a list?
        labels[a] = np.array(
            [concepts_with_label[1][attribute_var_name] for concepts_with_label in concepts_with_labels],
            dtype=np.float32)

    return concepts, labels


def compile_labelled_concepts(samples_query, concept_var_name, attribute_var_name, attribute_values,
                              train_and_eval_transaction, predict_transaction, sampling_params):
    """
    Assumes the case that data is partitioned into 2 keyspaces, one for training and evaluation, and another for
    prediction on unseen data (with labels). Therefore this function draws training and evaluation samples from the
    same keyspace.
    :param attribute_values:
    :param samples_query: Query to use to select possible samples
    :param concept_var_name: The variable used for the example concepts within the `samples_query`
    :param attribute_var_name: The variable used for the samples' labels (attributes) within the `samples_query`
    :param train_and_eval_transaction: Transaction for the training/evaluation keyspace
    :param predict_transaction: Transaction for the unseen prediction keyspace
    :param sampling_params: The number of samples to find for each of train, eval, predict, and the population sizes
    to pick from
    :return: dicts of concepts and labels with keys: 'train', 'eval', 'predict'
    """

    print(f'Finding concepts and labels')
    print('    for training and evaluation')
    concepts_dicts, labels_dicts = \
        query_for_random_samples_with_attribute(train_and_eval_transaction, samples_query,
                                                concept_var_name, attribute_var_name, attribute_values,
                                                sampling_params['train']['sample_size'] +
                                                sampling_params['eval']['sample_size'],
                                                sampling_params['train']['population_size'] +
                                                sampling_params['eval']['population_size'])
    print('    for prediction')
    concepts_dicts_predict, labels_dicts_predict = \
        query_for_random_samples_with_attribute(predict_transaction,
                                                samples_query,
                                                concept_var_name,
                                                attribute_var_name, attribute_values,
                                                sampling_params['predict']['sample_size'],
                                                sampling_params['predict']['population_size'])

    division = sampling_params['train']['sample_size']
    # Iterate over the classes
    concepts = {'train': [], 'eval': [], 'predict': []}
    labels = {'train': [], 'eval': [], 'predict': []}
    for label_value in concepts_dicts.keys():
        concepts['train'].extend(concepts_dicts[label_value][:division])
        concepts['eval'].extend(concepts_dicts[label_value][division:])
        labels['train'].extend(labels_dicts[label_value][:division])
        labels['eval'].extend(labels_dicts[label_value][division:])

        concepts['predict'].extend(concepts_dicts_predict[label_value])
        labels['predict'].extend(labels_dicts_predict[label_value])

    return concepts, labels


def delete_all_labels_from_keyspaces(keyspaces_transactions, attribute_type):
    # Warn the use this will delete concepts
    for keyspace_key, tx in keyspaces_transactions.items():
        # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
        # that we are certain that they aren't being used by the learner
        print(f'Deleting concepts from keyspace {keyspace_key} to avoid data pollution')
        delete_query = f'match $x isa {attribute_type}; delete $x;'
        print(delete_query)
        tx.query(delete_query)
        tx.commit()


def check_concepts_are_unique(concepts):
    concept_ids = [concept.id for concept in concepts]
    print(concept_ids)
    diff = len(concept_ids) - len(set(concept_ids))
    if diff != 0:
        raise ValueError(f'There are {diff} duplicate concepts present')