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

from kglib.kgcn_experimental.diagnosis.data import create_concept_graphs, write_predictions_to_grakn, get_all_types, \
    CATEGORICAL_ATTRIBUTES
from kglib.kgcn_experimental.graph_utils.iterate import multidigraph_node_data_iterator, multidigraph_data_iterator
from kglib.kgcn_experimental.graph_utils.prepare import apply_logits_to_graphs, duplicate_edges_in_reverse
from kglib.kgcn_experimental.network.attribute import CategoricalAttribute, BlankAttribute
from kglib.kgcn_experimental.network.model import softmax, KGCN
from kglib.kgcn_experimental.network.process import KGCNProcessor
from kglib.synthetic_graphs.diagnosis.main import generate_example_graphs


def diagnosis_example(num_graphs=60,
                      num_processing_steps_tr=10,
                      num_processing_steps_ge=10,
                      num_training_iterations=1000,
                      categorical_attributes=CATEGORICAL_ATTRIBUTES):

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

    # Encode attribute values
    for graph in concept_graphs:

        for data in multidigraph_data_iterator(graph):
            data['encoded_value'] = 0

        for node_data in multidigraph_node_data_iterator(graph):
            typ = node_data['type']

            # Add the integer value of the category for each categorical attribute instance
            for attr_typ, category_values in categorical_attributes.items():
                if typ == attr_typ:
                    node_data['encoded_value'] = category_values.index(node_data['value'])

    ind_concept_graphs = [nx.convert_node_labels_to_integers(graph, label_attribute='concept') for graph in concept_graphs]
    concept_graphs = [duplicate_edges_in_reverse(graph) for graph in ind_concept_graphs]

    tr_graphs = concept_graphs[:tr_ge_split]
    ge_graphs = concept_graphs[tr_ge_split:]

    type_embedding_dim = 5
    attr_embedding_dim = 6

    type_categories_list = [i for i, _ in enumerate(all_node_types)]
    non_attribute_nodes = type_categories_list.copy()

    attr_embedders = dict()

    # Construct categorical attribute embedders
    for attr_typ, category_values in categorical_attributes.items():
        num_categories = len(category_values)

        def make_blank_embedder():
            return CategoricalAttribute(num_categories, attr_embedding_dim, name=attr_typ + '_cat_embedder')
        attr_typ_index = all_node_types.index(attr_typ)

        # Record the embedder, and the index of the type that it should encode
        attr_embedders[make_blank_embedder] = [attr_typ_index]

        non_attribute_nodes.pop(attr_typ_index)

    # All entities and relations (non-attributes) also need an embedder with matching output dimension, which does
    # nothing. This is provided as a list of their indices
    def make_blank_embedder():
        return BlankAttribute(attr_embedding_dim)
    attr_embedders[make_blank_embedder] = non_attribute_nodes

    kgcn = KGCN(len(all_node_types),
                len(all_edge_types),
                type_embedding_dim,
                attr_embedding_dim,
                attr_embedders,
                edge_output_size=3,
                node_output_size=3)

    processor = KGCNProcessor(kgcn, all_node_types, all_edge_types)

    train_values, test_values = processor(tr_graphs,
                                          ge_graphs,
                                          num_processing_steps_tr=num_processing_steps_tr,
                                          num_processing_steps_ge=num_processing_steps_ge,
                                          num_training_iterations=num_training_iterations,
                                          log_every_seconds=2)

    logit_graphs = graphs_tuple_to_networkxs(test_values["outputs"][-1])

    ind_ge_graphs = ind_concept_graphs[tr_ge_split:]
    ge_graphs = apply_logits_to_graphs(ind_ge_graphs, logit_graphs)

    for ge_graph in ge_graphs:
        for data in multidigraph_data_iterator(ge_graph):
            data['probabilities'] = softmax(data['logits'])
            data['prediction'] = int(np.argmax(data['probabilities']))

    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()


if __name__ == "__main__":
    diagnosis_example()
