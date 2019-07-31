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

import random

from kglib.kgcn_experimental.genealogy.data import create_concept_graphs
from kglib.kgcn_experimental.model import model


def main():

    graph_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # The value at which to split the data into training and evaluation sets
    tr_ge_split = int(len(graph_ids)/2)

    random.seed(0)
    random.shuffle(graph_ids)
    print(f'Graphs used: {graph_ids}')
    all_node_types = ['person', 'parentship', 'grandparentship', 'siblingship']
    all_edge_types = ['parent', 'child', 'grandparent', 'grandchild', 'sibling']

    concept_graphs = create_concept_graphs(graph_ids)
    model(concept_graphs,
          all_node_types,
          all_edge_types,
          tr_ge_split,
          num_processing_steps_tr=10,
          num_processing_steps_ge=10,
          num_training_iterations=10000,
          log_every_seconds=2)


if __name__ == "__main__":
    main()
