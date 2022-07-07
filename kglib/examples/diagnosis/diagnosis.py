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
from torch import as_tensor
import torch.nn.functional as functional
import torch_geometric.transforms as transforms
from torch.nn import Linear
from torch_geometric.nn import HANConv
from typedb.client import *

from kglib.kgcn_data_loader.dataset.typedb_networkx_dataset import TypeDBNetworkxDataSet
from kglib.kgcn_data_loader.transform.binary_link_prediction import LinkPredictionLabeller, binary_relations_to_edges, \
    prepare_edge_triplets, prepare_node_types
from kglib.kgcn_data_loader.transform.typedb_graph_encoder import GraphFeatureEncoder, CategoricalEncoder, \
    ContinuousEncoder
from kglib.kgcn_data_loader.utils import load_typeql_schema_file, load_typeql_data_file
from kglib.utils.graph.iterate import multidigraph_node_data_iterator, multidigraph_edge_data_iterator
from kglib.utils.graph.query.query_graph import QueryGraph
from kglib.utils.typedb.synthetic.examples.diagnosis.generate import generate_example_data

DATABASE = "diagnosis"
ADDRESS = "localhost:1729"

PREEXISTS = 0

# Ignore any types that exist in the TypeDB instance but which aren't being used for learning to reduce the
# number of categories to embed
TYPES_TO_IGNORE = {'risk-factor', 'person-id', 'probability-preexists', 'probability-exists', 'probability-non-exists',
                   'alcohol-risked-disease', 'person-at-alcohol-risk', 'person-at-hereditary-risk',
                   'hereditary-risked-disease', 'smoking-risked-disease', 'person-at-smoking-risk',
                   'person-at-age-risk', 'age-risked-disease'}
# Note that this determines the edge direction when converting from a TypeDB relation
RELATION_TYPE_TO_PREDICT = ('person', 'patient', 'diagnosis', 'diagnosed-disease', 'disease')

# The learner should see candidate relations the same as the ground truth relations, so adjust these candidates to
# look like their ground truth counterparts
TYPES_AND_ROLES_TO_OBFUSCATE = {}


def diagnosis_example(typedb_binary_directory,
                      num_graphs,
                      database=DATABASE,
                      address=ADDRESS,
                      # TODO: remove hard-coding
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

    def clear_unneeded_fields(graph):
        for node_data in multidigraph_node_data_iterator(graph):
            x = node_data["x"]
            # y = node_data["y"]
            t = node_data["type"]
            node_data.clear()
            node_data["x"] = x
            # node_data["y"] = y
            node_data["type"] = t

        for edge_data in multidigraph_edge_data_iterator(graph):
            x = edge_data["edge_attr"]
            y = edge_data["y_edge"]
            t = edge_data["type"]
            edge_data.clear()
            edge_data["edge_attr"] = x
            edge_data["y_edge"] = y
            edge_data["type"] = t
        return graph

    edge_types = list({triplet[1] for triplet in edge_type_triplets})
    transform = transforms.Compose([
        prepare_graph,
        GraphFeatureEncoder(node_types, edge_types, attribute_encoders, attribute_encoding_size),
        LinkPredictionLabeller(RELATION_TYPE_TO_PREDICT[2]),
        clear_unneeded_fields
    ])

    # Create a Dataset that will load graphs from TypeDB on-demand, based on an ID
    dataset = TypeDBNetworkxDataSet(
        [0], node_types, edge_type_triplets, get_query_handles, DATABASE, ADDRESS, session, True, transform
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, node_type_indices, edge_type_indices = dataset[0]
    data = data.to_heterogeneous(
        as_tensor(node_type_indices), as_tensor(edge_type_indices), node_types, edge_type_triplets
    ).to(device)  # Get the first graph object.

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
        neg_sampling_ratio=1.0,
        # edge_types=edge_type_triplets,  # Evaluates to: ('person', 'diagnosis', 'disease'),
        # rev_edge_types=edge_type_triplets_reversed,  # Evaluates to: ('disease', 'rev_diagnosis', 'person'),
        edge_types=RELATION_TYPE_TO_PREDICT[::2],  # Evaluates to: ('person', 'diagnosis', 'disease'),
        rev_edge_types=edge_type_triplets_reversed[edge_type_triplets.index(RELATION_TYPE_TO_PREDICT[::2])]  # Evaluates to: ('disease', 'rev_diagnosis', 'person'),
    )(data)

    class HAN(torch.nn.Module):
        def __init__(self, in_channels: Union[int, Dict[str, int]],
                     out_channels: int, hidden_channels=128, heads=8):
            super().__init__()
            self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                    dropout=0.6, metadata=train_data.metadata())
            self.lin1 = Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = Linear(hidden_channels, out_channels)

        def forward(self, x_dict, edge_index_dict):
            out = self.han_conv(x_dict, edge_index_dict)
            # out = self.lin(out['person', 'disease'])
            row, col = edge_index_dict[('person', 'diagnosis', 'disease')]
            z = torch.cat([out['person'][row], out['disease'][col]], dim=-1)
            z = self.lin1(z).relu()
            z = self.lin2(z)
            # return z.view(-1)
            return z

    model = HAN(in_channels=-1, out_channels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    with torch.no_grad():  # Initialize lazy modules.
        out = model(train_data.x_dict, train_data.edge_index_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train() -> float:
        model.train()
        optimizer.zero_grad()
        # TODO: Does the model automatically use `edge_label_index_dict` and `edge_label` for training to include the negative samples? It seems like it can't be
        #  in which case we want to use the below, but it fails because `edge_label_index_dict` contains only one edge's information
        # pred = model(train_data.x_dict, train_data.edge_label_index_dict)
        # loss = functional.cross_entropy(pred, train_data[('person', 'diagnosis', 'disease')].edge_label)
        pred = model(train_data.x_dict, train_data.edge_index_dict)
        loss = functional.cross_entropy(pred, train_data[('person', 'diagnosis', 'disease')].y_edge)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test() -> List[float]:
        model.eval()
        accs = []
        for split in train_data, val_data, test_data:
            # We use `edge_index_dict` and `y_edge` for validation and testing to exclude the negative samples
            pred = model(split.x_dict, split.edge_index_dict).argmax(dim=-1)
            acc = (pred == split['person', 'disease'].y_edge).sum() / split['person', 'disease'].y_edge.numel()
            accs.append(float(acc))
        return accs

    best_val_acc = 0
    start_patience = patience = 100
    for epoch in range(1, 200):
        loss = train()
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
              f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

        if best_val_acc <= val_acc:
            patience = start_patience
            best_val_acc = val_acc
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

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

    return [
        (symptom_query, lambda x: x, symptom_query_graph),
        (diagnosis_query, lambda x: x, diagnosis_query_graph),
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
                             f'insert '
                             f'$pd(patient: $p, diagnosed-disease: $d) isa diagnosis,'
                             f'has probability-exists {p[2]:.3f},'
                             f'has probability-non-exists {p[1]:.3f},'  
                             f'has probability-preexists {p[0]:.3f};')
                    tx.query().insert(query)
    tx.commit()


if __name__ == '__main__':
    # TODO: Remove
    diagnosis_example("/Users/jamesfletcher/programming/typedb-dists/typedb-all-mac-2.11.0", 50)