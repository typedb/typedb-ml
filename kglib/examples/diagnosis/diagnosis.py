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

import inspect
import subprocess as sp

import networkx as nx
import torch
import torch.nn.functional as functional
import torch_geometric.transforms as transforms
from torch.nn import Linear
from torch_geometric.nn import SAGEConv, to_hetero
from typedb.client import *

from kglib.kgcn_data_loader.dataset.typedb_networkx_dataset import TypeDBNetworkxDataSet
from kglib.kgcn_data_loader.transform.binary_link_prediction import LinkPredictionLabeller, binary_relations_to_edges, \
    prepare_edge_triplets, prepare_node_types
from kglib.kgcn_data_loader.transform.typedb_graph_encoder import GraphFeatureEncoder, CategoricalEncoder, \
    ContinuousEncoder
from kglib.kgcn_data_loader.utils import load_typeql_schema_file, load_typeql_data_file
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.typedb.synthetic.examples.diagnosis.generate import generate_example_data

DATABASE = "diagnosis"
ADDRESS = "localhost:1729"

PREEXISTS = 0

# Ignore any types that exist in the TypeDB instance but which aren't being used for learning to reduce the
# number of categories to embed
TYPES_TO_IGNORE = {}

# Note that this determines the edge direction when converting from a TypeDB relation
RELATION_TYPE_TO_PREDICT = ('person', 'patient', 'diagnosis', 'diagnosed-disease', 'disease')

# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts
TYPES_AND_ROLES_TO_OBFUSCATE = {}


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

    # Delete the database each time  # TODO: Remove
    sp.check_call([
        './typedb',
        'console',
        f'--command=database delete {database}',
    ], cwd=typedb_binary_directory)

    client = TypeDB.core_client(address)
    create_database(client, database)

    load_typeql_schema_file(database, typedb_binary_directory, schema_file_path)
    load_typeql_data_file(database, typedb_binary_directory, seed_data_file_path)
    generate_example_data(client, num_graphs, database=database)

    session = client.session(database, SessionType.DATA)

    # During the transforms below we convert the *relations to predict* to simple edges, which means the relation
    # changes from a node to an edge. We therefore need to update the node_types and edge_types accordingly
    node_types = prepare_node_types(session, RELATION_TYPE_TO_PREDICT, TYPES_TO_IGNORE)
    edge_type_triplets, edge_type_triplets_reversed = prepare_edge_triplets(session, RELATION_TYPE_TO_PREDICT, TYPES_TO_IGNORE)

    # Attribute encoders encode the value of each attribute into a fixed-length feature vector. The encoders are
    # defined on a per-type basis. Easily define your own encoders for specific attribute data in your TypeDB database
    attribute_encoding_size = 16
    attribute_encoders = {
        # Categorical Attribute types and the values of their categories
        # TODO: Use a sentence encoder for this instead to demonstrate how to use one
        'name': CategoricalEncoder(
            ['Diabetes Type II', 'Multiple Sclerosis', 'Blurred vision', 'Fatigue', 'Cigarettes', 'Alcohol']
        ),
        # Continuous Attribute types and their min and max values
        'severity': ContinuousEncoder(0, 1),
        'age': ContinuousEncoder(7, 80),
        'units-per-week': ContinuousEncoder(3, 29)
    }

    def prepare_graph(graph):
        # TODO: We likely need to know the relations that were replaced with binary edges later on
        replaced_edges = binary_relations_to_edges(graph, RELATION_TYPE_TO_PREDICT[1:4]),
        return nx.convert_node_labels_to_integers(graph, label_attribute="concept")

    edge_types = list({triplet[1] for triplet in edge_type_triplets})
    transform = transforms.Compose([
        prepare_graph,
        GraphFeatureEncoder(node_types, edge_types, attribute_encoders, attribute_encoding_size),
        LinkPredictionLabeller(RELATION_TYPE_TO_PREDICT[2])
    ])
    # TODO: Consider clearing unneeded node and edge data

    # Create a Dataset that will load graphs from TypeDB on-demand, based on an ID
    dataset = TypeDBNetworkxDataSet([0], get_query_handles, DATABASE, ADDRESS, session, True, transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)  # Get the first graph object.

    # TODO: Consider whether we want to add reverse edges for the edge type being predicted and whether to do the
    #  before or after other transforms
    data = transforms.ToUndirected()(data)
    for edge_from, edge, edge_to in edge_type_triplets_reversed:
        # TODO: It's unclear in the PyG examples why this is necessary
        del data[edge_from, edge, edge_to].edge_label  # Remove "reverse" label.

    # TODO: How can the negative sampling know what types to add to the generated edges? It surely can't, so we have to
    #  deal with this
    train_data, val_data, test_data = transforms.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        add_negative_train_samples=True,
        neg_sampling_ratio=1.0,
        edge_types=edge_type_triplets,
        rev_edge_types=edge_type_triplets_reversed,
    )(data)

    class GNNEncoder(torch.nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    class EdgeDecoder(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, 1)

        def forward(self, z_dict, edge_label_index):
            row, col = edge_label_index
            z = torch.cat([z_dict['user'][row], z_dict['movie'][col]], dim=-1)

            z = self.lin1(z).relu()
            z = self.lin2(z)
            return z.view(-1)

    class Model(torch.nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='sum')
            self.decoder = EdgeDecoder(hidden_channels)

        def forward(self, x_dict, edge_index_dict, edge_label_index):
            z_dict = self.encoder(x_dict, edge_index_dict)
            return self.decoder(z_dict, edge_label_index)

    model = Model(hidden_channels=32).to(device)

    # Due to lazy initialization, we need to run one model step so the number
    # of parameters can be inferred:
    with torch.no_grad():
        model.encoder(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def weighted_mse_loss(pred, target, weight=None):
        weight = 1. if weight is None else weight[target].to(pred.dtype)
        return (weight * (pred - target.to(pred.dtype)).pow(2)).mean()

    def train():
        model.train()
        optimizer.zero_grad()
        pred = model(train_data.x_dict, train_data.edge_index_dict,
                     train_data['user', 'movie'].edge_label_index)
        target = train_data['user', 'movie'].edge_label
        loss = weighted_mse_loss(pred, target, None)  # TODO: Consider using weighting to balance the positive/negative example sets
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test(data):
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict,
                     data['user', 'movie'].edge_label_index)
        pred = pred.clamp(min=0, max=5)
        target = data['user', 'movie'].edge_label.float()
        rmse = functional.mse_loss(pred, target).sqrt()
        return float(rmse)

    for epoch in range(1, 301):
        loss = train()
        train_rmse = test(train_data)
        val_rmse = test(val_data)
        test_rmse = test(test_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, '
              f'Val: {val_rmse:.4f}, Test: {test_rmse:.4f}')

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


if __name__ == '__main__':
    # TODO: Remove
    diagnosis_example("/Users/jamesfletcher/programming/typedb-dists/typedb-all-mac-2.11.0", num_graphs=6)
