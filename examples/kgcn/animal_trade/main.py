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

import grakn
import tensorflow as tf

import kglib.kgcn.management.grakn as grakn_mgmt
import kglib.kgcn.management.logging as logging
import kglib.kgcn.management.persistence as prs
import kglib.kgcn.management.samples as samp_mgmt
import kglib.kgcn.models.downstream as downstream
import kglib.kgcn.models.model as model
import kglib.kgcn.neighbourhood.data.sampling.random_sampling as random_sampling
import kglib.kgcn.preprocess.persistence as persistence

flags = tf.app.flags
FLAGS = flags.FLAGS

# Learning params
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

# Sample selection params
EXAMPLES_QUERY = 'match $e(exchanged-item: $traded-item) isa exchange, has appendix $appendix; $appendix {}; get;'
LABEL_ATTRIBUTE_TYPE = 'appendix'
ATTRIBUTE_VALUES = [1, 2, 3]
EXAMPLE_CONCEPT_TYPE = 'traded-item'

NUM_PER_CLASS = 10
POPULATION_SIZE_PER_CLASS = 1000

# Params for persisting to files
DIR = os.path.dirname(os.path.realpath(__file__))
TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
BASE_PATH = f'{DIR}/dataset/{NUM_PER_CLASS}_concepts/'
flags.DEFINE_string('log_dir', BASE_PATH + 'out/out_' + TIMESTAMP, 'directory to use to store data from training')

SAVED_LABELS_PATH = BASE_PATH + 'labels/labels_{}.p'

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'

KEYSPACES = {
    TRAIN: "animaltrade_train",
    EVAL: "animaltrade_train",
    PREDICT: "animaltrade_test"
}

URI = "localhost:48555"

NEIGHBOUR_SAMPLE_SIZES = (7, 2, 2)

sys.stdout = logging.Logger(FLAGS.log_dir + '/output.log')


def main(modes=(TRAIN, EVAL, PREDICT)):

    client = grakn.Grakn(uri=URI)
    sessions = grakn_mgmt.get_sessions(client, KEYSPACES)
    transactions = grakn_mgmt.get_transactions(sessions)

    batch_size = NUM_PER_CLASS * FLAGS.num_classes
    kgcn = model.KGCN(NEIGHBOUR_SAMPLE_SIZES,
                      FLAGS.features_length,
                      FLAGS.starting_concepts_features_length,
                      FLAGS.aggregated_length,
                      FLAGS.output_length,
                      transactions[TRAIN],
                      batch_size,
                      sampling_method=random_sampling.random_sample,
                      sampling_limit_factor=4)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
    classifier = downstream.SupervisedKGCNClassifier(kgcn, optimizer, FLAGS.num_classes, FLAGS.log_dir,
                                                     max_training_steps=FLAGS.max_training_steps)

    feed_dicts = {}
    feed_dict_storer = persistence.FeedDictStorer(BASE_PATH + 'input/')

    # Overwrites any saved data
    try:
        for mode in modes:
            feed_dicts[mode] = feed_dict_storer.retrieve_feed_dict(mode)

    except FileNotFoundError:

        # Check if saved concepts and labels exist, and act accordingly
        # if check_for_saved_labelled_concepts(SAVED_LABELS_PATH, modes):
        try:
            concepts, labels = prs.load_saved_labelled_concepts(KEYSPACES, transactions, SAVED_LABELS_PATH)
        except FileNotFoundError:
            sampling_params = {
                TRAIN: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
                EVAL: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
                PREDICT: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
            }
            concepts, labels = samp_mgmt.compile_labelled_concepts(EXAMPLES_QUERY, EXAMPLE_CONCEPT_TYPE,
                                                                   LABEL_ATTRIBUTE_TYPE, ATTRIBUTE_VALUES,
                                                                   transactions[TRAIN], transactions[PREDICT],
                                                                   sampling_params)
            prs.save_labelled_concepts(KEYSPACES, concepts, labels, SAVED_LABELS_PATH)

            samp_mgmt.delete_all_labels_from_keyspaces(transactions, LABEL_ATTRIBUTE_TYPE)

            # Get new transactions since deleting labels requires committing and therefore closes transactions
            transactions = grakn_mgmt.get_transactions(sessions)
            # We need to re-fetch the sample concepts, since we need live transactions where the labels are removed
            concepts, labels = prs.load_saved_labelled_concepts(KEYSPACES, transactions, SAVED_LABELS_PATH)

        for mode in modes:
            feed_dicts[mode] = classifier.get_feed_dict(sessions[mode], concepts[mode], labels=labels[mode])
            feed_dict_storer.store_feed_dict(mode, feed_dicts[mode])

    # Train
    if TRAIN in modes:
        print("\n\n********** TRAIN Keyspace **********")
        classifier.train(feed_dicts[TRAIN])

    # Eval
    if EVAL in modes:
        print("\n\n********** EVAL Keyspace **********")
        # Presently, eval keyspace is the same as the TRAIN keyspace
        classifier.eval(feed_dicts[EVAL])

    # Predict
    if PREDICT in modes:
        print("\n\n********** PREDICT Keyspace **********")
        # We're using unseen data, but since we have labels we can use classifier.eval rather than classifier.predict
        classifier.eval(feed_dicts[PREDICT])

    grakn_mgmt.close(sessions)
    grakn_mgmt.close(transactions)


if __name__ == "__main__":
    main()
