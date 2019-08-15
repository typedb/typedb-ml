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
import copy
import inspect

import networkx as nx

from kglib.graph.create.from_queries import build_graph_from_queries
from kglib.graph.label.label import label_nodes_by_property, label_edges_by_property


def write_predictions_to_grakn(graphs, tx):
    """
    Take predictions from the ML model, and insert representations of those predictions back into the graph.

    Args:
        graphs: graphs containing the concepts, with their class predictions and class probabilities
        tx: Grakn write transaction to use

    Returns: None

    """
    for graph in graphs:
        for node, data in graph.nodes(data=True):
            if data['prediction'] == 2:
                concept = data['concept']
                concept_type = concept.type_label
                if concept_type == 'diagnosis' or concept_type == 'candidate-diagnosis':
                    neighbours = graph.neighbors(node)

                    for neighbour in neighbours:
                        concept = graph.nodes[neighbour]['concept']
                        if concept.type_label == 'person':
                            person = concept
                        else:
                            disease = concept

                    p = data['probabilities']
                    query = (f'match'
                             f'$p id {person.id};'
                             f'$d id {disease.id};'
                             f'insert'
                             f'$pd(predicted-patient: $p, predicted-diagnosed-disease: $d) isa predicted-diagnosis,'
                             f'has probability-exists {p[2]:.3f},'
                             f'has probability-non-exists {p[1]:.3f},'
                             f'has probability-preexists {p[0]:.3f};')
                    tx.query(query)
    tx.commit()


def create_concept_graphs(example_indices, grakn_session):
    graphs = []
    qh = QueryHandler()

    infer = True

    for example_id in example_indices:
        print(f'Creating graph for example {example_id}')
        graph_query_handles = qh.get_query_handles(example_id)
        with grakn_session.transaction().read() as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx, infer=infer)

        # Remove label leakage - change type labels that indicate candidates into non-candidates
        label_nodes_by_property(graph, 'type', 'candidate-diagnosis', {'type': 'diagnosis'})
        label_edges_by_property(graph, 'type', 'candidate-patient', {'type': 'patient'})
        label_edges_by_property(graph, 'type', 'candidate-diagnosed-disease', {'type': 'diagnosed-disease'})

        graph.name = example_id
        graphs.append(graph)

    return graphs


class QueryHandler:

    def __init__(self):
        # Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to
        # exist
        self.existing = dict(input=1, solution=0)

        # Elements to infer are the graph elements whose existence we want to predict to be true, they are positive
        # examples
        self.to_infer = dict(input=0, solution=2)

        # Candidates are neither present in the input nor in the solution, they are negative examples
        self.candidate = dict(input=0, solution=1)

    def diagnosis_query(self, example_id):
        return inspect.cleandoc(f'''match
               $p isa person, has example-id {example_id};
               $s isa symptom;
               $d isa disease;
               $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation;
               $c(cause: $d, effect: $s) isa causality;
               $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
               get;''')

    def base_query_graph(self):
        g = nx.MultiDiGraph()
        g.add_node('p', **self.existing)
        g.add_node('s', **self.existing)
        g.add_node('d', **self.existing)
        g.add_node('sp', **self.existing)
        g.add_edge('sp', 's', type='presented-symptom', **self.existing)
        g.add_edge('sp', 'p', type='symptomatic-patient', **self.existing)
        g.add_node('c', **self.existing)
        g.add_edge('c', 's', type='effect', **self.existing)
        g.add_edge('c', 'd', type='cause', **self.existing)
        return g

    def diagnosis_query_graph(self):
        g = self.base_query_graph()
        g = copy.copy(g)
        g.add_node('diag', **self.to_infer)
        g.add_edge('diag', 'd', type='diagnosed-disease', **self.to_infer)
        g.add_edge('diag', 'p', type='patient', **self.to_infer)
        return g

    def candidate_diagnosis_query(self, example_id):
        return inspect.cleandoc(f'''match
               $p isa person, has example-id {example_id};
               $s isa symptom;
               $d isa disease;
               $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation;
               $c(cause: $d, effect: $s) isa causality;
               $diag(candidate-patient: $p, candidate-diagnosed-disease: $d) isa candidate-diagnosis; 
               get;''')

    def candidate_diagnosis_query_graph(self):
        g = self.base_query_graph()
        g = copy.copy(g)
        g.add_node('diag', **self.candidate)
        g.add_edge('diag', 'd', type='candidate-diagnosed-disease', **self.candidate)
        g.add_edge('diag', 'p', type='candidate-patient', **self.candidate)
        return g

    def get_query_handles(self, example_id):

        return [
            (self.diagnosis_query(example_id), lambda x: x, self.diagnosis_query_graph()),
            (self.candidate_diagnosis_query(example_id), lambda x: x, self.candidate_diagnosis_query_graph()),
        ]
