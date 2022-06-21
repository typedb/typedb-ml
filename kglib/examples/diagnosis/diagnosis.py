#
#  Copyright (C) 2022 Vaticle
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

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as functional
import torch_geometric.transforms as transforms
import inspect
import subprocess as sp
import networkx as nx
from typedb.client import *

from kglib.kgcn_data_loader.dataset.typedb_networkx_dataset import TypeDBNetworkxDataSet
from kglib.kgcn_data_loader.transform.standard_kgcn_transform import StandardKGCNNetworkxTransform
from kglib.kgcn_data_loader.utils import load_typeql_schema_file, load_typeql_data_file
from kglib.utils.graph.iterate import multidigraph_data_iterator, multidigraph_node_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.graph.thing.queries_to_networkx_graph import build_graph_from_queries
from kglib.utils.typedb.synthetic.examples.diagnosis.generate import generate_example_data
from kglib.utils.typedb.type.type import get_thing_types, get_role_types, get_role_triples, get_has_triples

DATABASE = "diagnosis"
ADDRESS = "localhost:1729"

PREEXISTS = 0

# Categorical Attribute types and the values of their categories
CATEGORICAL_ATTRIBUTES = {
    'name': [
        'Diabetes Type II', 'Multiple Sclerosis', 'Blurred vision', 'Fatigue', 'Cigarettes', 'Alcohol'
    ]
}
# Continuous Attribute types and their min and max values
CONTINUOUS_ATTRIBUTES = {'severity': (0, 1), 'age': (7, 80), 'units-per-week': (3, 29)}

TYPES_TO_IGNORE = []
ROLES_TO_IGNORE = []

# Note that this determines the edge direction when converting from a TypeDB relation
RELATION_TYPE_TO_PREDICT = ('patient', 'diagnosis', 'diagnosed-disease')

# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts
TYPES_AND_ROLES_TO_OBFUSCATE = {}


def binary_relations_to_edges(graph: nx.MultiDiGraph, binary_relation_type):
    """
    A function to convert a TypeDB relation (a hyperedge) to a directed binary edge.

    The replaced relations may not be:
    - a roleplayer in any other relations
    - own any attributes
    - have more than two roles
    and must:
    - have exactly two outgoing role edges (which can be of the same role)

    This is useful because edges are often stored as an adjacency matrix. In PyTorch Geometric, for example,
    this adjacency matrix expects binary edges. To be compatible we convert to binary edges so that when creating
    negative samples we can simply change the values of the adjacency matrix. In the case of a hyperedge we cannot
    simply do this to add negative edges.

    Args:
        graph: The graph to modify in-place
        binary_relation_type: A triple of the `(role1, relation, role2)` types to convert to a single edge labelled with `relation`

    Returns:
        Dict of edges to the node each replaces
    """
    replacement_edges = {}
    for node, node_data in graph.nodes(data=True):
        if node_data["type"] == binary_relation_type[1]:
            if len(list(graph.in_edges(node))) > 0:
                raise ValueError(
                    "The given binary relation can't be transformed into an edge because it plays a role in another "
                    "relation."
                )
            roles = list(graph.edges(node, data=True))  # equivalent to out_edges()
            assert len(roles) == 2
            new_edge_start = None
            new_edge_end = None
            for from_roleplayer, to_roleplayer, data in roles:
                if data["type"] == binary_relation_type[0]:
                    if new_edge_start is None:
                        new_edge_start = to_roleplayer
                    else:
                        # In this case the given relation type is symmetric. Therefore direction is arbitrary (but
                        # expect to add a reverse edge elsewhere).
                        assert binary_relation_type[0] == binary_relation_type[2]
                        new_edge_end = to_roleplayer
                elif data["type"] == binary_relation_type[2]:
                    new_edge_end = to_roleplayer
                else:
                    raise ValueError(
                        f"Unexpected role in relation {binary_relation_type[1]}. Expected \""
                        f"{binary_relation_type[0]}\" or \"{binary_relation_type[2]}\" but got \"{data['type']}\"."
                    )
                graph.remove_edge(from_roleplayer, to_roleplayer)
            graph.add_edge(new_edge_start, new_edge_end, type=binary_relation_type[1])
            replacement_edges[(new_edge_start, new_edge_end)] = node
    for node in replacement_edges.values():
        graph.remove_node(node)
    return replacement_edges


def get_types(session, types_to_ignore, roles_to_ignore):
    with session.transaction(TransactionType.READ) as tx:
        # The terminology changes from here onwards from thing -> node and role -> edge
        node_types = get_thing_types(tx)
        edge_types = get_role_types(tx)
        # Ignore any types or roles that exist in the TypeDB instance but which aren't being used for learning to
        # reduce the number of categories to embed
        [node_types.remove(el) for el in types_to_ignore]
        [edge_types.remove(el) for el in roles_to_ignore]
        print(f'Found node types: {node_types}')
        print(f'Found edge types: {edge_types}')
        return node_types, edge_types


def get_edge_types(session):
    # TODO: Naming is too close to get_types where the result is very different
    with session.transaction(TransactionType.READ) as tx:
        edge_types = get_role_triples(tx) + get_has_triples(tx)
    return edge_types


def reverse_edge_types(edge_types):
    reversed = []
    for edge_from, edge, edge_to in edge_types:
        reversed.append((edge_to, f"rev_{edge}", edge_from))
        return reversed


def diagnosis_example(typedb_binary_directory,
                      num_graphs=100,
                      database=DATABASE,
                      address=ADDRESS,
                      schema_file_path="/Users/jamesfletcher/programming/research/kglib/utils/typedb/synthetic/examples/diagnosis/schema.tql",
                      seed_data_file_path="/Users/jamesfletcher/programming/research/kglib/utils/typedb/synthetic/examples/diagnosis/seed_data.tql"):
    """
    Run the diagnosis example from start to finish, including traceably ingesting predictions back into TypeDB

    Args:
        typedb_binary_directory: Location of the typedb binary for the purpose of loading initial schema and data
        num_graphs: Number of graphs to use for training and testing combined
        database: The name of the database to retrieve example subgraphs from
        address: The address of the running TypeDB instance
        schema_file_path: Path to the diagnosis schema file
        seed_data_file_path: Path to the file containing seed data, that doesn't grow as synthetic data is added

    Returns:
        Final accuracies for training and for testing
    """

    # Delete the database each time
    sp.check_call([
        './typedb',
        'console',
        f'--command=database delete {database}',
    ], cwd=typedb_binary_directory)

    tr_ge_split = int(num_graphs*0.5)

    client = TypeDB.core_client(address)
    create_database(client, database)

    load_typeql_schema_file(database, typedb_binary_directory, schema_file_path)
    load_typeql_data_file(database, typedb_binary_directory, seed_data_file_path)
    generate_example_data(client, num_graphs, database=database)

    session = client.session(database, SessionType.DATA)

    node_types, edge_types = get_types(session, TYPES_TO_IGNORE, ROLES_TO_IGNORE)

    # TODO: Here temporarily, should be graph-by-graph
    graph_query_handles = get_query_handles(0)
    options = TypeDBOptions.core()
    options.infer = True
    with session.transaction(TransactionType.READ, options) as tx:
        # Build a graph from the queries, samplers, and query graphs
        graph = build_graph_from_queries(graph_query_handles, tx)

    binary_relations_to_edges(graph, RELATION_TYPE_TO_PREDICT)  # TODO: Should be a transform. Do we do this before or after generating features?

    # This transform adds the features and labels to the graph
    # TODO: We should make the feature encoders more explicit
    transform = StandardKGCNNetworkxTransform(
        node_types,
        edge_types,
        target_name='solution',  # TODO: We're planning to do away with already having the graphs labelled at this
        # point, so somehow we need to add negative samples and label accordingly
        obfuscate=None,
        categorical=None,
        continuous=None,
        duplicate_in_reverse=True,
        label_attribute="concept",
    )

    # Create a Dataset that will load graphs from TypeDB on-demand, based on an ID
    dataset = TypeDBNetworkxDataSet(
        list(range(num_graphs)),
        get_query_handles,
        DATABASE,
        ADDRESS,
        session,
        True,
        transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)  # Get the first graph object.

    edge_types = get_edge_types(session)
    edge_types_reversed = reverse_edge_types(edge_types)

    data = transforms.ToUndirected()(data)  # TODO: Consider whether we want to add reverse edges for the edge type being predicted
    for edge_from, edge, edge_to in edge_types_reversed:
        del data[edge_from, edge, edge_to].edge_label  # Remove "reverse" label.

    # TODO: How can the negative sampling know what types to add to the generated edges? It surely can't, so we have to
    #  deal with this
    train_data, val_data, test_data = transforms.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=edge_types,
        rev_edge_types=edge_types_reversed,
    )(data)

    # TODO: Replace this model with that from hetero_link_pred
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            torch.manual_seed(1234567)
            # self.conv1 = GCNConv(dataset.num_features, hidden_channels)
            # self.conv2 = GCNConv(hidden_channels, dataset.num_classes)
            self.conv1 = GCNConv(3, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, 3)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = x.relu()
            x = functional.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x

    model = GCN(hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        out = model(data.x, data.edge_index)  # Perform a single forward pass.
        # loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
        loss = criterion(out, data.y)  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        return loss

    def test():
        model.eval()
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        # test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
        test_correct = pred == data.y  # Check against ground-truth labels.
        # test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
        test_acc = int(test_correct.sum()) / int(data.sum())  # Derive ratio of correct predictions.
        return test_acc

    # for epoch in range(1, 101):
    #     loss = train()
    #     print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

    with session.transaction(TransactionType.WRITE) as tx:
        write_predictions_to_typedb(ge_graphs, tx)

    session.close()
    client.close()

    return solveds_tr, solveds_ge


def create_database(client, database):
    if client.databases().contains(database):
        raise ValueError(
            f"There is already a database present with the name {database}. The Diagnosis example expects a clean DB. "
            f"Please delete the {database} database, or use another database name")
    client.databases().create(database)


def create_concept_graphs(example_indices, typedb_session, infer = True):
    """
    Builds an in-memory graph for each example, with an example_id as an anchor for each example subgraph.
    Args:
        example_indices: The values used to anchor the subgraph queries within the entire knowledge graph
        typedb_session: TypeDB Session

    Returns:
        In-memory graphs of TypeDB subgraphs
    """

    graphs = []

    options = TypeDBOptions.core()
    options.infer = infer

    for example_id in example_indices:
        print(f'Creating graph for example {example_id}')
        graph_query_handles = get_query_handles(example_id)

        with typedb_session.transaction(TransactionType.READ, options) as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx)

        obfuscate_labels(graph, TYPES_AND_ROLES_TO_OBFUSCATE)

        graph.name = example_id
        graphs.append(graph)

    return graphs


def obfuscate_labels(graph, types_and_roles_to_obfuscate):
    # Remove label leakage - change type labels that indicate candidates into non-candidates
    for data in multidigraph_data_iterator(graph):
        for label_to_obfuscate, with_label in types_and_roles_to_obfuscate.items():
            if data['type'] == label_to_obfuscate:
                data.update(type=with_label)
                break


def get_query_handles(example_id):
    """
    Creates an iterable, each element containing a TypeQL query, a function to sample the answers, and a QueryGraph
    object which must be the TypeDB graph representation of the query. This tuple is termed a "query_handle"

    Args:
        example_id: A uniquely identifiable attribute value used to anchor the results of the queries to a specific
                    subgraph

    Returns:
        query handles
    """
    assert example_id == 0

    # === Hereditary Feature ===
    hereditary_query = inspect.cleandoc(f'''match
           $p isa person;
           $par isa person;
           $ps(child: $p, parent: $par) isa parentship;
           $diag(patient:$par, diagnosed-disease: $d) isa diagnosis;
           $d isa disease, has name $n;
          ''')

    vars = p, par, ps, d, diag, n = 'p', 'par', 'ps', 'd', 'diag', 'n'
    hereditary_query_graph = (QueryGraph()
                              .add_vars(vars, PREEXISTS)
                              .add_role_edge(ps, p, 'child', PREEXISTS)
                              .add_role_edge(ps, par, 'parent', PREEXISTS)
                              .add_role_edge(diag, par, 'patient', PREEXISTS)
                              .add_role_edge(diag, d, 'diagnosed-disease', PREEXISTS)
                              .add_has_edge(d, n, PREEXISTS))

    # === Consumption Feature ===
    consumption_query = inspect.cleandoc(f'''match
           $p isa person;
           $s isa substance, has name $n;
           $c(consumer: $p, consumed-substance: $s) isa consumption, 
           has units-per-week $u;''')

    vars = p, s, n, c, u = 'p', 's', 'n', 'c', 'u'
    consumption_query_graph = (QueryGraph()
                               .add_vars(vars, PREEXISTS)
                               .add_has_edge(s, n, PREEXISTS)
                               .add_role_edge(c, p, 'consumer', PREEXISTS)
                               .add_role_edge(c, s, 'consumed-substance', PREEXISTS)
                               .add_has_edge(c, u, PREEXISTS))

    # === Age Feature ===
    person_age_query = inspect.cleandoc(f'''match 
            $p isa person, has age $a; 
           ''')

    vars = p, a = 'p', 'a'
    person_age_query_graph = (QueryGraph()
                              .add_vars(vars, PREEXISTS)
                              .add_has_edge(p, a, PREEXISTS))

    # === Risk Factors Feature ===
    risk_factor_query = inspect.cleandoc(f'''match 
            $d isa disease; 
            $p isa person; 
            $r(person-at-risk: $p, risked-disease: $d) isa risk-factor; 
           ''')

    vars = p, d, r = 'p', 'd', 'r'
    risk_factor_query_graph = (QueryGraph()
                               .add_vars(vars, PREEXISTS)
                               .add_role_edge(r, p, 'person-at-risk', PREEXISTS)
                               .add_role_edge(r, d, 'risked-disease', PREEXISTS))

    # === Symptom ===
    vars = p, s, sn, d, dn, sp, sev, c = 'p', 's', 'sn', 'd', 'dn', 'sp', 'sev', 'c'

    symptom_query = inspect.cleandoc(f'''match
           $p isa person;
           $s isa symptom, has name $sn;
           $d isa disease, has name $dn;
           $sp(presented-symptom: $s, symptomatic-patient: $p) isa symptom-presentation, has severity $sev;
           $c(cause: $d, effect: $s) isa causality;
          ''')

    symptom_query_graph = (QueryGraph()
                           .add_vars(vars, PREEXISTS)
                           .add_has_edge(s, sn, PREEXISTS)
                           .add_has_edge(d, dn, PREEXISTS)
                           .add_role_edge(sp, s, 'presented-symptom', PREEXISTS)
                           .add_has_edge(sp, sev, PREEXISTS)
                           .add_role_edge(sp, p, 'symptomatic-patient', PREEXISTS)
                           .add_role_edge(c, s, 'effect', PREEXISTS)
                           .add_role_edge(c, d, 'cause', PREEXISTS))

    # === Diagnosis ===

    diag, d, p, dn = 'diag', 'd', 'p', 'dn'

    diagnosis_query = inspect.cleandoc(f'''match
           $p isa person;
           $d isa disease, has name $dn;
           $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
          ''')

    diagnosis_query_graph = (QueryGraph()
                             .add_vars([diag], PREEXISTS)
                             .add_vars([d, p, dn], PREEXISTS)
                             .add_role_edge(diag, d, 'diagnosed-disease', PREEXISTS)
                             .add_role_edge(diag, p, 'patient', PREEXISTS))

    # === Candidate Diagnosis ===
    candidate_diagnosis_query = inspect.cleandoc(f'''match
           $p isa person;
           $d isa disease, has name $dn;
           $diag(candidate-patient: $p, candidate-diagnosed-disease: $d) isa candidate-diagnosis; 
          ''')

    candidate_diagnosis_query_graph = (QueryGraph()
                                       .add_vars([diag], PREEXISTS)
                                       .add_vars([d, p, dn], PREEXISTS)
                                       .add_role_edge(diag, d, 'candidate-diagnosed-disease', PREEXISTS)
                                       .add_role_edge(diag, p, 'candidate-patient', PREEXISTS))

    return [
        (symptom_query, lambda x: x, symptom_query_graph),
        (diagnosis_query, lambda x: x, diagnosis_query_graph),
        (candidate_diagnosis_query, lambda x: x, candidate_diagnosis_query_graph),
        (risk_factor_query, lambda x: x, risk_factor_query_graph),
        (person_age_query, lambda x: x, person_age_query_graph),
        (consumption_query, lambda x: x, consumption_query_graph),
        (hereditary_query, lambda x: x, hereditary_query_graph)
    ]


def write_predictions_to_typedb(graphs, tx):
    """
    Take predictions from the ML model, and insert representations of those predictions back into the graph.

    Args:
        graphs: graphs containing the concepts, with their class predictions and class probabilities
        tx: TypeDB write transaction to use

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
                    query = (f'match '
                             f'$p iid {person.iid};'
                             f'$d iid {disease.iid};'
                             f'$kgcn isa kgcn;'
                             f'insert '
                             f'$pd(patient: $p, diagnosed-disease: $d, diagnoser: $kgcn) isa diagnosis,'
                             f'has probability-exists {p[2]:.3f},'
                             f'has probability-non-exists {p[1]:.3f},'  
                             f'has probability-preexists {p[0]:.3f};')
                    tx.query().insert(query)
    tx.commit()
