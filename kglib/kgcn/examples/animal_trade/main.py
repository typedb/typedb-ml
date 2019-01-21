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

import os
import sys
import time

import collections
import grakn
import numpy as np
import tensorflow as tf

import kglib.kgcn.models.downstream as downstream
import kglib.kgcn.models.model as model
import kglib.kgcn.neighbourhood.data.sampling.random_sampling as random
import kglib.kgcn.preprocess.persistence as persistence
import kglib.kgcn.use_cases.attribute_prediction.label_extraction as label_extraction

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('num_classes', 3, 'Number of classes')
flags.DEFINE_integer('features_length', 198, 'Number of features after encoding')
flags.DEFINE_integer('starting_concepts_features_length', 173,
                     'Number of features after encoding for the nodes of interest, which excludes the features for '
                     'role_type and role_direction')
flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
                                          'and the final length of the embeddings')

flags.DEFINE_integer('max_training_steps', 10000, 'Max number of gradient steps to take during gradient descent')

TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")

NUM_PER_CLASS = 30
POPULATION_SIZE_PER_CLASS = 1000
# BASE_PATH = f'dataset/{NUM_PER_CLASS}_concepts/'
BASE_PATH = 'dataset/30_concepts_ordered_7x2x2_without_attributes_with_rules/'
flags.DEFINE_string('log_dir', BASE_PATH + 'out/out_' + TIMESTAMP, 'directory to use to store data from training')

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'


def query_for_labelled_examples(tx, sample_size, population_size, offset):
    appendix_vals = [1, 2, 3]

    concepts = {}
    labels = {}

    for a in appendix_vals:
        target_concept_query = f'match $e(exchanged-item: $t) isa exchange, has appendix $appendix; $appendix {a}; ' \
                               f'offset {offset}; limit {population_size}; get;'

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           ('t', collections.OrderedDict([('appendix', appendix_vals)])),
                                                           sampling_method=random.random_sample
                                                           )
        concepts_with_labels = extractor(tx, sample_size)
        if len(concepts_with_labels) == 0:
            raise RuntimeError(f'Couldn\'t find any concepts to match target query "{target_concept_query}"')

        concepts[a] = [concepts_with_label[0] for concepts_with_label in concepts_with_labels]
        # TODO Should this be an array not a list?
        labels[a] = np.array([concepts_with_label[1]['appendix'] for concepts_with_label in concepts_with_labels], dtype=np.float32)

    return concepts, labels


def combine_labelled_examples(concepts_dict, labels_dict):
    concepts = []
    labels = []
    for key in concepts_dict.keys():
        concepts.extend(concepts_dict[key])
        labels.extend(labels_dict[key])
    return concepts, labels


def retrieve_persisted_labelled_examples(tx, file_path):

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


def check_concepts_are_unique(concepts):
    concept_ids = [concept.id for concept in concepts]
    print(concept_ids)
    diff = len(concept_ids) - len(set(concept_ids))
    if diff != 0:
        raise ValueError(f'There are {diff} duplicate concepts present')


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


sys.stdout = Logger(FLAGS.log_dir + '/output.log')


def main():
    keyspaces = {TRAIN: "animaltrade_train",
                 EVAL: "animaltrade_train",
                 PREDICT: "animaltrade_test"
                 }
    uri = "localhost:48555"
    sessions = {}
    txs = {}

    client = grakn.Grakn(uri=uri)

    labels_file_root = BASE_PATH + 'labels/labels_{}.p'

    find_labelled_concepts = False
    delete_all_labels_from_keyspace = False
    run = True
    save_input_data = False  # Overwrites any saved data

    concepts = {}
    labels = {}

    for keyspace_key in list(keyspaces.keys()):
        sessions[keyspace_key] = client.session(keyspace=keyspaces[keyspace_key])
        txs[keyspace_key] = sessions[keyspace_key].transaction(grakn.TxType.WRITE)

    if find_labelled_concepts:
        print(f'Finding concepts and labels')
        print('    for training and evaluation')
        concepts_dicts, labels_dicts = query_for_labelled_examples(txs[TRAIN], NUM_PER_CLASS * 2, POPULATION_SIZE_PER_CLASS*2, 0)

        half = NUM_PER_CLASS
        # Iterate over the classes
        concepts[TRAIN] = []
        concepts[EVAL] = []
        labels[TRAIN] = []
        labels[EVAL] = []
        for key in concepts_dicts.keys():
            concepts[TRAIN].extend(concepts_dicts[key][:half])
            concepts[EVAL].extend(concepts_dicts[key][half:])
            labels[TRAIN].extend(labels_dicts[key][:half])
            labels[EVAL].extend(labels_dicts[key][half:])

        print('    for prediction')
        concepts_dicts_predict, labels_dicts_predict = query_for_labelled_examples(txs[PREDICT], NUM_PER_CLASS,
                                                                                   POPULATION_SIZE_PER_CLASS, 0)
        concepts[PREDICT], labels[PREDICT] = combine_labelled_examples(concepts_dicts_predict, labels_dicts_predict)

        for keyspace_key in list(keyspaces.keys()):
            persistence.save_variable(([concept.id for concept in concepts[keyspace_key]], labels[keyspace_key]),
                                      labels_file_root.format(keyspace_key))

        # if delete_all_labels_from_keyspace:
        #     # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
        #     # that we are certain that they aren't being used by the learner
        #     print('Deleting concepts to avoid pollution')
        #     txs[keyspace_key].query(f'match $x isa appendix; delete $x;')
        #     txs[keyspace_key].commit()

    else:
        for keyspace_key in list(keyspaces.keys()):
            print(f'Loading concepts and labels for {keyspace_key}')
            concepts[keyspace_key], labels[keyspace_key] = retrieve_persisted_labelled_examples(txs[keyspace_key],
                                                                                                labels_file_root.format(
                                                                                                    keyspace_key))

    feed_dict_storer = persistence.FeedDictStorer(BASE_PATH + 'input/')

    if (not find_labelled_concepts) and run:
        # neighbour_sample_sizes = (8, 2, 4)
        neighbour_sample_sizes = (7, 2, 2)  # TODO (7, 2, 2) Throws an error without rules

        # storage_path=BASE_PATH+'input/'

        batch_size = buffer_size = NUM_PER_CLASS * FLAGS.num_classes
        kgcn = model.KGCN(neighbour_sample_sizes,
                          FLAGS.features_length,
                          FLAGS.starting_concepts_features_length,
                          FLAGS.aggregated_length,
                          FLAGS.output_length,
                          txs[TRAIN],
                          batch_size,
                          buffer_size,
                          # sampling_method=random_sampling.random_sample,
                          # sampling_limit_factor=4
                          )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        classifier = downstream.SupervisedKGCNClassifier(kgcn, optimizer, FLAGS.num_classes, FLAGS.log_dir,
                                                         max_training_steps=FLAGS.max_training_steps)

        feed_dict = {}

        if save_input_data:

            feed_dict[TRAIN] = classifier.get_feed_dict(sessions[TRAIN], concepts[TRAIN], labels=labels[TRAIN])
            feed_dict_storer.store_feed_dict(TRAIN, feed_dict[TRAIN])

            feed_dict[EVAL] = classifier.get_feed_dict(sessions[EVAL], concepts[EVAL], labels=labels[EVAL])
            feed_dict_storer.store_feed_dict(EVAL, feed_dict[EVAL])

        else:
            feed_dict[TRAIN] = feed_dict_storer.retrieve_feed_dict(TRAIN)
            feed_dict[EVAL] = feed_dict_storer.retrieve_feed_dict(EVAL)

        # Train
        classifier.train(feed_dict[TRAIN])

        # Eval
        classifier.eval(feed_dict[EVAL])

    for keyspace_key in list(keyspaces.keys()):
        # Close all transactions to clean up
        txs[keyspace_key].close()
        sessions[keyspace_key].close()


if __name__ == "__main__":
    main()
