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

import numpy as np

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