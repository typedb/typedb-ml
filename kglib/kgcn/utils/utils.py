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
import os
import sys

import grakn
import numpy as np

from kglib.kgcn.examples.animal_trade.main import NUM_PER_CLASS
from kglib.kgcn.neighbourhood.data.sampling import random_sampling as random
from kglib.kgcn.preprocess import persistence as persistence
from kglib.kgcn.use_cases.attribute_prediction import label_extraction as label_extraction


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def check_concepts_are_unique(concepts):
    concept_ids = [concept.id for concept in concepts]
    print(concept_ids)
    diff = len(concept_ids) - len(set(concept_ids))
    if diff != 0:
        raise ValueError(f'There are {diff} duplicate concepts present')


def retrieve_persisted_labelled_concepts(tx, file_path):

    concept_ids, labels = persistence.load_variable(file_path)
    print('==============================')
    print('Loaded concept IDs with labels')
    [print(concept_id, label) for concept_id, label in zip(concept_ids, labels)]
    concepts = []
    for concept_id in concept_ids:
        query = f'match $x id {concept_id}; get;'
        concept = next(tx.query(query)).get('x')
        concepts.append(concept)

    return concepts, labels


def query_for_random_examples_with_attribute(tx, query, example_var_name, attribute_var_name, attribute_vals,
                                             sample_size_per_label, population_size):
    concepts = {}
    labels = {}

    for a in attribute_vals:
        target_concept_query = query.format(a, population_size)

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           (example_var_name, collections.OrderedDict(
                                                               [(attribute_var_name, attribute_vals)])),
                                                           sampling_method=random.random_sample
                                                           )
        concepts_with_labels = extractor(tx, sample_size_per_label)
        if len(concepts_with_labels) == 0:
            raise RuntimeError(f'Couldn\'t find any concepts to match target query "{target_concept_query}"')

        concepts[a] = [concepts_with_label[0] for concepts_with_label in concepts_with_labels]
        # TODO Should this be an array not a list?
        labels[a] = np.array(
            [concepts_with_label[1][attribute_var_name] for concepts_with_label in concepts_with_labels],
            dtype=np.float32)

    return concepts, labels


def delete_all_labels_from_keyspaces(keyspaces_transactions, attribute_type):
    # Warn the use this will delete concepts
    while True:
        ans = input(
            f'This operation will delete all concepts of type {attribute_type} in the given keyspaces. Proceed? y/n')

        if ans not in ['y', 'Y', 'n', 'N']:
            print('please enter y or n.')
            continue
        if ans == 'y' or ans == 'Y':
            for tx in keyspaces_transactions:
                # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
                # that we are certain that they aren't being used by the learner
                print('Deleting concepts to avoid data pollution')
                tx.query(f'match $x isa {attribute_type}; delete $x;')
                tx.commit()
            break
        if ans == 'n' or ans == 'N':
            return False


def load_saved_labelled_concepts(keyspaces, transactions, saved_labels_path):
    concepts = {}
    labels = {}
    for keyspace_key in list(keyspaces.keys()):
        print(f'Loading concepts and labels for {keyspace_key}')
        concepts[keyspace_key], labels[keyspace_key] = retrieve_persisted_labelled_concepts(
            transactions[keyspace_key], saved_labels_path.format(keyspace_key))
    return concepts, labels


def get_sessions(client, keyspaces):
    sessions = {}
    for keyspace_key, keyspace_name in keyspaces:
        sessions[keyspace_key] = client.session(keyspace=keyspace_name)
    return sessions


def get_transactions(sessions):
    transactions = {}
    for keyspace_key, session in sessions.items():
        transactions[keyspace_key] = session.transaction(grakn.TxType.WRITE)
    return transactions


def close(to_close: dict):
    for val in to_close.values():
        val.close()


def save_labelled_concepts(keyspaces, concepts, labels, saved_labels_path):
    for keyspace_key in list(keyspaces.keys()):
        persistence.save_variable(([concept.id for concept in concepts[keyspace_key]], labels[keyspace_key]),
                                  saved_labels_path.format(keyspace_key))


def compile_labelled_concepts(examples_query, concept_var_name, attribute_var_name, train_and_eval_transaction,
                              predict_transaction, sampling_params):
    """
    Assumes the case that data is partitioned into 2 keyspaces, one for training and evaluation, and another for
    prediction on unseen data (with labels). Therefore this function draws training and evaluation examples from the
    same keyspace.
    :param examples_query: Query to use to select possible examples
    :param concept_var_name: The variable used for the example concepts within the `examples_query`
    :param attribute_var_name: The variable used for the examples' labels (attributes) within the `examples_query`
    :param train_and_eval_transaction: Transaction for the training/evaluation keyspace
    :param predict_transaction: Transaction for the unseen prediction keyspace
    :param sampling_params: The number of examples to find for each of train, eval, predict, and the population sizes
    to pick from
    :return: dicts of concepts and labels with keys: 'train', 'eval', 'predict'
    """

    print(f'Finding concepts and labels')
    print('    for training and evaluation')
    concepts_dicts, labels_dicts = \
        utils.query_for_random_examples_with_attribute(train_and_eval_transaction, examples_query,
                                                       concept_var_name, attribute_var_name, [1, 2, 3],
                                                       sampling_params['train']['sample_size'] +
                                                       sampling_params['eval']['sample_size'],
                                                       sampling_params['train']['population_size'] +
                                                       sampling_params['eval']['population_size'])
    print('    for prediction')
    concepts_dicts_predict, labels_dicts_predict = \
        utils.query_for_random_examples_with_attribute(predict_transaction,
                                                       examples_query,
                                                       concept_var_name,
                                                       attribute_var_name, [1, 2, 3],
                                                       sampling_params['predict']['sample_size'],
                                                       sampling_params['predict']['population_size'])

    half = NUM_PER_CLASS
    # Iterate over the classes
    concepts = {'train': [], 'eval': [], 'predict': []}
    labels = {'train': [], 'eval': [], 'predict': []}
    for label_value in concepts_dicts.keys():
        concepts['train'].extend(concepts_dicts[label_value][:half])
        concepts['eval'].extend(concepts_dicts[label_value][half:])
        labels['train'].extend(labels_dicts[label_value][:half])
        labels['eval'].extend(labels_dicts[label_value][half:])

        concepts['predict'].extend(concepts_dicts_predict[label_value])
        labels['predict'].extend(labels_dicts_predict[label_value])

    return concepts, labels