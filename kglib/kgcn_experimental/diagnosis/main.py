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

import networkx as nx
import numpy as np
from grakn.client import GraknClient
from graph_nets.utils_np import graphs_tuple_to_networkxs

from kglib.kgcn_experimental.diagnosis.data import create_concept_graphs, write_predictions_to_grakn, get_all_types
from kglib.kgcn_experimental.model import model, softmax
from kglib.kgcn_experimental.prepare import apply_logits_to_graphs, duplicate_edges_in_reverse
from kglib.synthetic_graphs.diagnosis.main import generate_example_graphs


def main(num_graphs=60, num_processing_steps_tr=10, num_processing_steps_ge=10, num_training_iterations=1000):

    # The value at which to split the data into training and evaluation sets
    tr_ge_split = int(num_graphs*0.5)
    keyspace = "diagnosis"
    uri = "localhost:48555"

    generate_example_graphs(num_graphs, keyspace=keyspace, uri=uri)

    # Get the node and edge types, these can simply be passed as a list, but here we query the schema and then remove
    # unwanted types and roles

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    with session.transaction().read() as tx:
        all_node_types, all_edge_types = get_all_types(tx)

    concept_graphs = create_concept_graphs(list(range(num_graphs)), session)

    tr_graphs = concept_graphs[:tr_ge_split]
    ge_graphs = concept_graphs[tr_ge_split:]

    ind_tr_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in tr_graphs]
    ind_ge_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in ge_graphs]

    tr_graphs = [duplicate_edges_in_reverse(graph) for graph in ind_tr_graphs]
    ge_graphs = [duplicate_edges_in_reverse(graph) for graph in ind_ge_graphs]

    train_values, test_values = model(tr_graphs,
                                      ge_graphs,
                                      all_node_types,
                                      all_edge_types,
                                      num_processing_steps_tr=num_processing_steps_tr,
                                      num_processing_steps_ge=num_processing_steps_ge,
                                      num_training_iterations=num_training_iterations,
                                      log_every_seconds=2)

    logit_graphs = graphs_tuple_to_networkxs(test_values["outputs"][-1])
    ge_graphs = apply_logits_to_graphs(ind_ge_graphs, logit_graphs)

    for ge_graph in ge_graphs:
        for node, data in ge_graph.nodes(data=True):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

        for sender, receiver, keys, data in ge_graph.edges(keys=True, data=True):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()


if __name__ == "__main__":
    main()
