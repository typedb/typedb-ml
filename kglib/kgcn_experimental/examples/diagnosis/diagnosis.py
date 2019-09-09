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

from grakn.client import GraknClient

from kglib.kgcn_experimental.examples.diagnosis.template import template
from kglib.utils.grakn.synthetic.examples.diagnosis.generate import generate_example_graphs
from kglib.utils.graph.create.queries_to_graph import build_graph_from_queries
from kglib.utils.graph.create.query_graph import QueryGraph
from kglib.utils.graph.iterate import multidigraph_data_iterator


def diagnosis_example(num_graphs=60,
                      num_processing_steps_tr=10,
                      num_processing_steps_ge=10,
                      num_training_iterations=1000,
                      keyspace="diagnosis", uri="localhost:48555"):

    tr_ge_split = int(num_graphs*0.5)

    generate_example_graphs(num_graphs, keyspace=keyspace, uri=uri)

    client = GraknClient(uri=uri)
    session = client.session(keyspace=keyspace)

    graphs = create_concept_graphs(list(range(num_graphs)), session)

    with session.transaction().read() as tx:
        all_node_types, all_edge_types = get_all_types(tx)

    ge_graphs = template(graphs,
                         tr_ge_split,
                         all_node_types,
                         all_edge_types,
                         num_processing_steps_tr=num_processing_steps_tr,
                         num_processing_steps_ge=num_processing_steps_ge,
                         num_training_iterations=num_training_iterations,
                         categorical_attributes=CATEGORICAL_ATTRIBUTES,
                         )

    with session.transaction().write() as tx:
        write_predictions_to_grakn(ge_graphs, tx)

    session.close()
    client.close()


CATEGORICAL_ATTRIBUTES = {'name': ['meningitis', 'flu', 'fever', 'light-sensitivity']}


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
        for data in multidigraph_data_iterator(graph):
            typ = data['type']
            if typ == 'candidate-diagnosis':
                data.update(type='diagnosis')
            elif typ == 'candidate-patient':
                data.update(type='patient')
            elif typ == 'candidate-diagnosed-disease':
                data.update(type='diagnosed-disease')

        graph.name = example_id
        graphs.append(graph)

    return graphs


# Existing elements in the graph are those that pre-exist in the graph, and should be predicted to continue to exist
PREEXISTS = dict(input=1, solution=0)

# Elements to infer are the graph elements whose existence we want to predict to be true, they are positive examples
TO_INFER = dict(input=0, solution=2)

# Candidates are neither present in the input nor in the solution, they are negative examples
CANDIDATE = dict(input=0, solution=1)


class QueryHandler:

    def diagnosis_query(self, example_id):
        return inspect.cleandoc(f'''match
               $p isa person, has example-id {example_id};
               $s isa symptom, has name $sn;
               $d isa disease, has name $dn;
               $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation;
               $c(cause: $d, effect: $s) isa causality;
               $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
               get;''')

    def base_query_graph(self):
        p, s, sn, d, dn, sp, c = 'p', 's', 'sn', 'd', 'dn', 'sp', 'c'
        g = QueryGraph()
        g.add_vars(p, s, sn, d, dn, sp, c, **PREEXISTS)
        g.add_has_edge(s, sn, **PREEXISTS)
        g.add_has_edge(d, dn, **PREEXISTS)
        g.add_role_edge(sp, s, 'presented-symptom', **PREEXISTS)
        g.add_role_edge(sp, p, 'symptomatic-patient', **PREEXISTS)
        g.add_role_edge(c, s, 'effect', **PREEXISTS)
        g.add_role_edge(c, d, 'cause', **PREEXISTS)

        return g

    def diagnosis_query_graph(self):
        g = self.base_query_graph()
        g = copy.copy(g)

        diag, d, p = 'diag', 'd', 'p'
        g.add_vars(diag, **TO_INFER)
        g.add_role_edge(diag, d, 'diagnosed-disease', **TO_INFER)
        g.add_role_edge(diag, p, 'patient', **TO_INFER)

        return g

    def candidate_diagnosis_query(self, example_id):
        return inspect.cleandoc(f'''match
               $p isa person, has example-id {example_id};
               $s isa symptom, has name $sn;
               $d isa disease, has name $dn;
               $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation;
               $c(cause: $d, effect: $s) isa causality;
               $diag(candidate-patient: $p, candidate-diagnosed-disease: $d) isa candidate-diagnosis; 
               get;''')

    def candidate_diagnosis_query_graph(self):
        g = self.base_query_graph()
        g = copy.copy(g)

        diag, d, p = 'diag', 'd', 'p'
        g.add_vars(diag, **CANDIDATE)
        g.add_role_edge(diag, d, 'candidate-diagnosed-disease', **CANDIDATE)
        g.add_role_edge(diag, p, 'candidate-patient', **CANDIDATE)

        return g

    def get_query_handles(self, example_id):

        return [
            (self.diagnosis_query(example_id), lambda x: x, self.diagnosis_query_graph()),
            (self.candidate_diagnosis_query(example_id), lambda x: x, self.candidate_diagnosis_query_graph()),
        ]


def get_all_types(tx):
    schema_concepts = tx.query("match $x sub thing; get;").collect_concepts()
    all_node_types = [schema_concept.label() for schema_concept in schema_concepts]
    [all_node_types.remove(el) for el in
     ['thing', 'relation', 'entity', 'attribute', '@has-attribute', '@key-attribute', 'candidate-diagnosis',
      'example-id', '@key-example-id', '@key-name', '@has-probability-exists', '@has-probability-non-exists',
      '@has-probability-preexists', 'probability-exists', 'probability-non-exists', 'probability-preexists']]
    print(all_node_types)

    roles = tx.query("match $x sub role; get;").collect_concepts()
    all_edge_types = ['has'] + [role.label() for role in roles]
    [all_edge_types.remove(el) for el in
     ['role', '@has-attribute-value', '@has-attribute-owner', 'candidate-patient',
      'candidate-diagnosed-disease', '@key-example-id-value', '@key-example-id-owner',
      '@key-name-value', '@key-name-owner']]
    print(all_edge_types)
    return all_node_types, all_edge_types


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


if __name__ == "__main__":
    diagnosis_example()
