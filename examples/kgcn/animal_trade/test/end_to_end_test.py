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
import subprocess as sub
import time
import unittest

import grakn
import tensorflow as tf

import kglib.kgcn.management.grakn as grakn_mgmt
import kglib.kgcn.management.samples as samp_mgmt
import kglib.kgcn.models.downstream as downstream
import kglib.kgcn.models.model as model
import kglib.kgcn.neighbourhood.data.sampling.random_sampling as random_sampling

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
flags.DEFINE_integer('max_training_steps', 50, 'Max number of gradient steps to take during gradient descent')

# Sample selection params
EXAMPLES_QUERY = 'match $e(exchanged-item: $traded-item) isa exchange, has appendix $appendix; $appendix {}; get;'
LABEL_ATTRIBUTE_TYPE = 'appendix'
ATTRIBUTE_VALUES = [1, 2, 3]
EXAMPLE_CONCEPT_TYPE = 'traded-item'

NUM_PER_CLASS = 5
POPULATION_SIZE_PER_CLASS = 100

# Params for persisting to files
DIR = os.path.dirname(os.path.realpath(__file__))
TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'

KEYSPACES = {
    TRAIN: "animaltrade_train",
    EVAL: "animaltrade_train",
    PREDICT: "animaltrade_train",
}

URI = "localhost:48555"

NEIGHBOUR_SAMPLE_SIZES = (2, 1)


class TestEndToEnd(unittest.TestCase):

    def test_end_to_end(self):
        # Unzip the Grakn distribution containing our data
        sub.run(['unzip', 'external/animaltrade_dist/file/downloaded', '-d',
                          'external/animaltrade_dist/file/downloaded-unzipped'])

        # Start Grakn
        sub.run(['external/animaltrade_dist/file/downloaded-unzipped/grakn-animaltrade/grakn', 'server', 'start'])

        modes = (TRAIN, EVAL)

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
        classifier = downstream.SupervisedKGCNClassifier(kgcn, optimizer, FLAGS.num_classes, None,
                                                         max_training_steps=FLAGS.max_training_steps)

        feed_dicts = {}

        sampling_params = {
            TRAIN: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
            EVAL: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
            PREDICT: {'sample_size': NUM_PER_CLASS, 'population_size': POPULATION_SIZE_PER_CLASS},
        }
        concepts, labels = samp_mgmt.compile_labelled_concepts(EXAMPLES_QUERY, EXAMPLE_CONCEPT_TYPE,
                                                               LABEL_ATTRIBUTE_TYPE, ATTRIBUTE_VALUES,
                                                               transactions[TRAIN], transactions[PREDICT],
                                                               sampling_params)

        for mode in modes:
            feed_dicts[mode] = classifier.get_feed_dict(sessions[mode], concepts[mode], labels=labels[mode])

        # Note: The ground-truth attribute labels haven't been removed from Grakn, so the results found here are
        # invalid, and used as an end-to-end test only

        # Train
        if TRAIN in modes:
            print("\n\n********** TRAIN Keyspace **********")
            classifier.train(feed_dicts[TRAIN])

        # Eval
        if EVAL in modes:
            print("\n\n********** EVAL Keyspace **********")
            # Presently, eval keyspace is the same as the TRAIN keyspace
            classifier.eval(feed_dicts[EVAL])

        grakn_mgmt.close(sessions)
        grakn_mgmt.close(transactions)


if __name__ == "__main__":
    unittest.main()
