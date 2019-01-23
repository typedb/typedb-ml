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

from kglib.kgcn.preprocess import persistence as persistence


def load_saved_labelled_concepts(keyspaces, transactions, saved_labels_path):
    concepts = {}
    labels = {}
    for keyspace_key in list(keyspaces.keys()):
        print(f'Loading concepts and labels for {keyspace_key}')
        concepts[keyspace_key], labels[keyspace_key] = retrieve_persisted_labelled_concepts(
            transactions[keyspace_key], saved_labels_path.format(keyspace_key))
    return concepts, labels


def save_labelled_concepts(keyspaces, concepts, labels, saved_labels_path):
    for keyspace_key in list(keyspaces.keys()):
        persistence.save_variable(([concept.id for concept in concepts[keyspace_key]], labels[keyspace_key]),
                                  saved_labels_path.format(keyspace_key))


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
