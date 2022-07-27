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
import argparse
import inspect
import os

import networkx as nx
import torch
import torch.nn.functional as functional
import torch_geometric.transforms as transforms
from torch import as_tensor
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import HGTConv
from typedb.client import *

from examples.diagnosis.dataset.generate import generate_example_data
from typedb_ml.networkx.query_graph import QueryGraph, Query
from typedb_ml.pytorch_geometric.dataset.dataset import DataSet
from typedb_ml.pytorch_geometric.transform.binary_link_prediction import LinkPredictionLabeller, \
    binary_relations_to_edges, binary_link_prediction_edge_triplets
from typedb_ml.pytorch_geometric.transform.common import clear_unneeded_fields, store_concepts_by_type
from typedb_ml.pytorch_geometric.transform.encode import FeatureEncoder, CategoricalEncoder, ContinuousEncoder
from typedb_ml.typedb.load import load_typeql_file, FileType
from typedb_ml.typedb.type import get_thing_types

DATABASE = "diagnosis"
ADDRESS = "localhost:1729"

# Ignore any types that exist in the TypeDB instance but which aren't being used for learning to reduce the
# number of categories to embed
TYPES_TO_IGNORE = {'risk-factor', 'person-id', 'alcohol-risked-disease', 'person-at-alcohol-risk',
                   'person-at-hereditary-risk', 'hereditary-risked-disease', 'smoking-risked-disease',
                   'person-at-smoking-risk', 'person-at-age-risk', 'age-risked-disease', 'predicted-diagnosis'}
# Note that this determines the edge direction when converting from a TypeDB relation
RELATION_TYPE_TO_PREDICT = ('person', 'patient', 'diagnosis', 'diagnosed-disease', 'disease')

TYPE_ENCODING_SIZE = 16
ATTRIBUTE_ENCODING_SIZE = 16

# Attribute encoders encode the value of each attribute into a fixed-length feature vector. The encoders are
# defined on a per-type basis. Easily define your own encoders for specific attribute data in your TypeDB database
ATTRIBUTE_ENCODERS = {
    # Categorical Attribute types and the values of their categories
    'name': CategoricalEncoder(
        ['Diabetes Type II', 'Multiple Sclerosis', 'Blurred vision', 'Fatigue', 'Cigarettes', 'Alcohol'],
        ATTRIBUTE_ENCODING_SIZE
    ),
    # Continuous Attribute types and their min and max values
    'severity': ContinuousEncoder(0, 1, ATTRIBUTE_ENCODING_SIZE),
    'age': ContinuousEncoder(7, 80, ATTRIBUTE_ENCODING_SIZE),
    'units-per-week': ContinuousEncoder(3, 29, ATTRIBUTE_ENCODING_SIZE)
}


def diagnosis_example(typedb_binary_directory,
                      num_graphs,
                      database=DATABASE,
                      address=ADDRESS,
                      schema_file_path="examples/diagnosis/dataset/schema.tql",
                      seed_data_file_path="examples/diagnosis/dataset/seed_data.tql"):
    """
    Args:
        typedb_binary_directory: Location of the TypeDB binary for the purpose of loading initial schema and data
        num_graphs: Number of graphs to use for training and testing combined
        database: The name of the database to retrieve data from
        address: The address of the running TypeDB instance
        schema_file_path: Path to the diagnosis schema file
        seed_data_file_path: Path to the file containing seed data

    Returns:
        Final accuracies for training and for testing
    """

    client = TypeDB.core_client(address)
    create_database(client, database)

    load_typeql_file(typedb_binary_directory, database, schema_file_path, FileType.Schema)
    load_typeql_file(typedb_binary_directory, database, seed_data_file_path, FileType.Data)
    generate_example_data(client, num_graphs, database=database)

    session = client.session(database, SessionType.DATA)

    # During the transforms below we convert the relations to predict to simple binary edges, which means the relation
    # changes from a node to an edge. We therefore need to update the node_types and edge_types accordingly

    # Remove the relation from the node types, since we will be using it as a binary edge instead
    to_ignore = list(TYPES_TO_IGNORE) + [RELATION_TYPE_TO_PREDICT[2]]
    node_types = [t for t in get_thing_types(session) if t not in to_ignore]
    edge_type_triplets, edge_type_triplets_reversed = binary_link_prediction_edge_triplets(
        session, RELATION_TYPE_TO_PREDICT, TYPES_TO_IGNORE
    )

    binary_edge_to_predict = RELATION_TYPE_TO_PREDICT[::2]  # Evaluates to: ('person', 'diagnosis', 'disease')
    binary_rev_edge_to_predict = edge_type_triplets_reversed[edge_type_triplets.index(RELATION_TYPE_TO_PREDICT[::2])]  # Evaluates to: ('disease', 'rev_diagnosis', 'person')

    edge_types = list({triplet[1] for triplet in edge_type_triplets})
    transform = transforms.Compose([
        lambda graph: binary_relations_to_edges(graph, RELATION_TYPE_TO_PREDICT[1:4]),
        lambda graph: nx.convert_node_labels_to_integers(graph, label_attribute="concept"),
        FeatureEncoder(node_types, edge_types, TYPE_ENCODING_SIZE, ATTRIBUTE_ENCODERS, ATTRIBUTE_ENCODING_SIZE),
        LinkPredictionLabeller(RELATION_TYPE_TO_PREDICT[2]),
        store_concepts_by_type,
        clear_unneeded_fields
    ])

    # Create a Dataset that will load graphs from TypeDB on-demand, based on an ID
    dataset = DataSet([0], node_types, edge_type_triplets, build_queries, session, True, transform)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, node_type_indices, edge_type_indices = dataset[0]
    data = data.to_heterogeneous(
        as_tensor(node_type_indices), as_tensor(edge_type_indices), node_types, edge_type_triplets
    ).to(device)  # Get the first graph object.

    # Reverse edges need to be present for bi-directional message-passing but the labels should not be considered
    # for node and edge representations
    data = transforms.ToUndirected()(data)
    for edge_from, edge, edge_to in edge_type_triplets_reversed:
        del data[edge_from, edge, edge_to].edge_label  # Remove "reverse" label.

    # Setting the neg_sampling_ratio higher than the number of places that a negative sample can be added causes the
    # training set to have too few negative samples!
    # Consider using other samplers from Pytorch Geometric in place of this one, depending on your use case
    train_data, val_data, test_data = transforms.RandomLinkSplit(
        num_val=0.2,
        num_test=0.2,
        neg_sampling_ratio=1.0,
        edge_types=binary_edge_to_predict,
        rev_edge_types=binary_rev_edge_to_predict
    )(data)

    # Add a new `links` attribute to store the edges for prediction so that they are definitely isolated from training
    train_data.link_index = train_data.edge_label_index_dict[binary_edge_to_predict]
    train_data.link_labels = train_data.edge_label_dict[binary_edge_to_predict]
    val_data.link_index = val_data.edge_label_index_dict[binary_edge_to_predict]
    val_data.link_labels = val_data.edge_label_dict[binary_edge_to_predict]
    test_data.link_index = test_data.edge_label_index_dict[binary_edge_to_predict]
    test_data.link_labels = test_data.edge_label_dict[binary_edge_to_predict]

    # Delete the stores for the predicted edge now that we have stored it elsewhere above
    data.links = data[binary_edge_to_predict]
    data.rev_links = data[binary_rev_edge_to_predict]
    del data[binary_edge_to_predict]
    del data[binary_rev_edge_to_predict]
    del train_data[binary_edge_to_predict]
    del train_data[binary_rev_edge_to_predict]
    del val_data[binary_edge_to_predict]
    del val_data[binary_rev_edge_to_predict]
    del test_data[binary_edge_to_predict]
    del test_data[binary_rev_edge_to_predict]

    class LinkPredictionModel(torch.nn.Module):
        def __init__(self, in_channels: Union[int, Dict[str, int]], hidden_channels=128, heads=8):
            super().__init__()
            self.conv = HGTConv(in_channels, hidden_channels, heads=heads, metadata=train_data.metadata())

        def encode(self, x_dict, edge_index_dict):
            return self.conv(x_dict, edge_index_dict)

        def decode(self, z, edge_label_index_dict):
            row, col = edge_label_index_dict
            logits = (z['person'][row] * z['disease'][col]).sum(dim=-1)
            return logits

        def decode_all(self, z):
            logits = z['person'] @ z['disease'].t()
            return logits

    model = LinkPredictionModel(in_channels=-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, model = data.to(device), model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)

    def train() -> float:
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x_dict, train_data.edge_index_dict)
        logits = model.decode(z, train_data.link_index)
        loss = functional.binary_cross_entropy_with_logits(logits, train_data.link_labels)
        loss.backward()
        optimizer.step()
        return float(loss)

    @torch.no_grad()
    def test() -> List[Tuple[float, float, float]]:
        model.eval()
        results = []
        for split in train_data, val_data, test_data:
            # We use `edge_index_dict` and `y_edge` for validation and testing to exclude the negative samples
            z = model.encode(split.x_dict, split.edge_index_dict)
            link_logits = model.decode(z, split.link_index)
            link_probs = link_logits.sigmoid()
            tp = ((link_probs > 0.5) * (split.link_labels == 1)).sum()
            tn = ((link_probs < 0.5) * (split.link_labels == 0)).sum()
            pos = (split.link_labels == 1).sum()
            neg = (split.link_labels == 0).sum()
            precision = tn / neg
            recall = tp / pos
            acc = (tp + tn) / (pos + neg)
            results.append((float(acc), precision, recall))
        return results

    writer = SummaryWriter()
    for edge_type, edge_store in zip(data.edge_types, data.edge_stores):
        writer.add_histogram('('+', '.join(edge_type) + ')/edge_attr', edge_store["edge_attr"])
        writer.add_histogram('('+', '.join(edge_type) + ')/y_edge', edge_store["y_edge"])

    for node_type, node_store in zip(data.node_types, data.node_stores):
        writer.add_histogram(node_type + '/x', node_store["x"])

    best_val_acc = 0
    start_patience = patience = 100
    train_results = None
    test_results = None
    for epoch in range(1, 100):
        loss = train()
        writer.add_scalar('Loss/train', loss, epoch)
        train_results, val_results, test_results = test()
        writer.add_scalar('Accuracy/train', train_results[0], epoch)
        writer.add_scalar('Accuracy/val', val_results[0], epoch)
        writer.add_scalar('Accuracy/test', test_results[0], epoch)
        writer.add_scalar('Precision/test', test_results[1], epoch)
        writer.add_scalar('Recall/test', test_results[2], epoch)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_results[0]:.4f}, '
              f'Val: {val_results[0]:.4f}, Test: {test_results[0]:.4f}, Test Precision: {test_results[1]:.4f}, '
              f'Test Recall: {test_results[2]:.4f}')

        if best_val_acc <= val_results[0]:
            patience = start_patience
            best_val_acc = val_results[0]
        else:
            patience -= 1

        if patience <= 0:
            print('Stopping training as validation accuracy did not improve '
                  f'for {start_patience} epochs')
            break

    z = model.encode(data.x_dict, data.edge_index_dict)
    final_edge_index = (model.decode_all(z).sigmoid() > 0.5).nonzero(as_tuple=False).cpu().detach().numpy()

    # Get back the concepts for each of links predicted in order to insert the predictions into TypeDB
    predicted_links = []
    for p, d in final_edge_index:
        predicted_links.append(
            {
                'person': data['concepts_by_type']['person'][p],
                'disease': data['concepts_by_type']['disease'][d]
            }
        )
    print("The following links have been predicted:")
    print(predicted_links)  # Bear in mind this is predicted links across *all* data: train, val and test

    with session.transaction(TransactionType.WRITE) as tx:
        write_predictions_to_typedb(predicted_links, tx)

    # Now we can get the confusion matrix from querying TypeDB! Note that this includes training and validation
    # examples, but serves as a demo for seeing the predictions made.
    with session.transaction(TransactionType.READ) as tx:
        # Also try these queries in TypeDB Studio omitting "count;" to visualise the predicted relations
        tp = tx.query().match_aggregate("match $p isa person; $d isa disease; ($p, $d) isa diagnosis; "
                                        "($p, $d) isa predicted-diagnosis; count;").get().as_int()
        tn = tx.query().match_aggregate("match $p isa person; $d isa disease; not{($p, $d) isa diagnosis;}; "
                                        "not{($p, $d) isa predicted-diagnosis;}; count;").get().as_int()
        fp = tx.query().match_aggregate("match $p isa person; $d isa disease; not{($p, $d) isa diagnosis;}; "
                                        "($p, $d) isa predicted-diagnosis; count;").get().as_int()
        fn = tx.query().match_aggregate("match $p isa person; $d isa disease; ($p, $d) isa diagnosis; "
                                        "not{($p, $d) isa predicted-diagnosis;}; count;").get().as_int()
        print("Confusion matrix")
        print(f"{tp} {fn}\n{fp} {tn}")

    session.close()
    client.close()

    return train_results[0], test_results[0]


def create_database(client, database):
    if client.databases().contains(database):
        raise ValueError(
            f"There is already a database present with the name {database}. The Diagnosis example expects a clean DB. "
            f"Please delete the {database} database, or use another database name")
    client.databases().create(database)


def build_queries(subgraph_id: int) -> List[Query]:
    """
    Creates a tuple of Query objects that contain the information needed to convert query answers into NetworkX graphs.

    Args:
        subgraph_id: A uniquely identifiable id used to anchor the results of the queries to a specific subgraph,
        designed so that the user can easily query for segmented subgraphs to be used as batches.

    Returns:
        List of Query
    """
    assert subgraph_id == 0  # In this example the graph is small so we don't use any subgraphs

    # === Hereditary Feature ===
    hereditary_query = inspect.cleandoc(f'''match
           $p isa person;
           $par isa parent;
           $ps(child: $p, parent: $par) isa parentship;
           $diag(patient:$par, diagnosed-disease: $d) isa familial-diagnosis;
           $d isa disease, has name $n;
          ''')

    vars = p, par, ps, d, diag, n = 'p', 'par', 'ps', 'd', 'diag', 'n'
    hereditary_query_graph = (QueryGraph()
                              .add_vars(vars)
                              .add_role_edge(ps, p, 'child')
                              .add_role_edge(ps, par, 'parent')
                              .add_role_edge(diag, par, 'patient')
                              .add_role_edge(diag, d, 'diagnosed-disease')
                              .add_has_edge(d, n))

    # === Consumption Feature ===
    consumption_query = inspect.cleandoc(f'''match
           $p isa person;
           $s isa substance, has name $n;
           $c(consumer: $p, consumed-substance: $s) isa consumption, 
           has units-per-week $u;''')

    vars = p, s, n, c, u = 'p', 's', 'n', 'c', 'u'
    consumption_query_graph = (QueryGraph()
                               .add_vars(vars)
                               .add_has_edge(s, n)
                               .add_role_edge(c, p, 'consumer')
                               .add_role_edge(c, s, 'consumed-substance')
                               .add_has_edge(c, u))

    # === Age Feature ===
    person_age_query = inspect.cleandoc(f'''match 
            $p isa person, has age $a; 
           ''')

    vars = p, a = 'p', 'a'
    person_age_query_graph = (QueryGraph()
                              .add_vars(vars)
                              .add_has_edge(p, a))

    # === Risk Factors Feature ===
    risk_factor_query = inspect.cleandoc(f'''match 
            $d isa disease; 
            $p isa person; 
            $r(person-at-risk: $p, risked-disease: $d) isa risk-factor; 
           ''')

    vars = p, d, r = 'p', 'd', 'r'
    risk_factor_query_graph = (QueryGraph()
                               .add_vars(vars)
                               .add_role_edge(r, p, 'person-at-risk')
                               .add_role_edge(r, d, 'risked-disease'))

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
                           .add_vars(vars)
                           .add_has_edge(s, sn)
                           .add_has_edge(d, dn)
                           .add_role_edge(sp, s, 'presented-symptom')
                           .add_has_edge(sp, sev)
                           .add_role_edge(sp, p, 'symptomatic-patient')
                           .add_role_edge(c, s, 'effect')
                           .add_role_edge(c, d, 'cause'))

    # === Diagnosis ===

    diag, d, p, dn = 'diag', 'd', 'p', 'dn'

    diagnosis_query = inspect.cleandoc(f'''match
           $p isa person;
           $d isa disease, has name $dn;
           $diag(patient: $p, diagnosed-disease: $d) isa diagnosis;
          ''')

    diagnosis_query_graph = (QueryGraph()
                             .add_vars([diag])
                             .add_vars([d, p, dn])
                             .add_role_edge(diag, d, 'diagnosed-disease')
                             .add_role_edge(diag, p, 'patient'))

    return [
        Query(symptom_query_graph, symptom_query),
        Query(diagnosis_query_graph, diagnosis_query),
        Query(risk_factor_query_graph, risk_factor_query),
        Query(person_age_query_graph, person_age_query),
        Query(consumption_query_graph, consumption_query),
        Query(hereditary_query_graph, hereditary_query)
    ]


def write_predictions_to_typedb(predicted_links, tx):
    """
    Take predictions from the ML model, and insert representations of those predictions back into the graph.

    Args:
        predicted_links: pairs of concepts that are predicted links
        tx: TypeDB write transaction to use

    Returns: None

    """
    for predicted_link in predicted_links:
        person = predicted_link['person']
        disease = predicted_link['disease']
        query = (f'match '
                 f'$p iid {person.iid}; '
                 f'$d iid {disease.iid}; '
                 f'insert '
                 f'$pd(patient: $p, diagnosed-disease: $d) isa predicted-diagnosis;')
        tx.query().insert(query)
    tx.commit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Just an example",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--graphs", help="num graphs", default=200)
    parser.add_argument("typedb", help="TypeDB location")
    args = parser.parse_args()
    cwd = os.getcwd()
    diagnosis_example(args.typedb, args.graphs,
                      database=DATABASE,
                      address=ADDRESS,
                      schema_file_path=cwd + '/' + "examples/diagnosis/dataset/schema.tql",
                      seed_data_file_path=cwd + '/' + "examples/diagnosis/dataset/seed_data.tql")
