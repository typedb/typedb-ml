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

from typing import Sequence, Callable, Optional

import networkx as nx
from torch_geometric.utils import from_networkx
from typedb.client import TypeDB, TypeDBSession, SessionType, TypeDBOptions, TransactionType

from kglib.networkx.thing.queries_to_networkx_graph import build_graph_from_queries


class DataSet:
    """
    Loading graphs based on queries from TypeDB.
    Note: not dependent on PyTorch or Pytorch Geometric.
    """

    def __init__(
        self,
        indices: Sequence,
        node_types,
        edge_type_triplets,
        get_query_handles_for_id: Callable,
        database: Optional[str] = None,
        uri: Optional[str] = "localhost:1729",
        session: Optional[TypeDBSession] = None,
        infer: bool = True,
        transform: Optional[Callable[[nx.Graph], nx.Graph]] = None,
    ):
        assert (database and uri) or session
        self._indices = indices
        self._node_types = node_types
        self._edge_type_triplets = edge_type_triplets
        self.get_query_handles_for_id = get_query_handles_for_id
        self._infer = infer
        self._transform = transform
        self._uri = uri
        self._database = database
        self._typedb_session = session

    @property
    def typedb_session(self):
        """
        Did this like this in an attempt to make it
        also work when using with a DataLoader with
        num_workers > 0.

        TODO: it does not, so look into this.
        """
        if not self._typedb_session:
            print("setting up session")
            print(self)
            client = TypeDB.core_client(self._uri)
            self._typedb_session = client.session(database=self._database, session_type=SessionType.DATA)
        return self._typedb_session

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        print(type(self._typedb_session))
        id = self._indices[idx]
        print(f"Fetching graph for id: {id}")
        graph_query_handles = self.get_query_handles_for_id(id)

        options = TypeDBOptions.core()
        options.infer = self._infer

        with self.typedb_session.transaction(TransactionType.READ, options=options) as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx)
        graph.name = id
        if self._transform:
            graph = self._transform(graph)
        data = from_networkx(graph)
        data.concepts_by_type = graph.concepts_by_type
        return data, self.node_type_indices(graph), self.edge_type_indices(graph)

    def node_type_indices(self, graph):
        indices = []
        for _, data in graph.nodes(data=True):
            indices.append(self._node_types.index(data['type']))
        return indices

    def edge_type_indices(self, graph):
        indices = []
        for src, dst, data in graph.edges(data=True):
            # indices.append(self._edge_type_triplets.index(data['type']))
            indices.append(self._edge_type_triplets.index((graph.nodes[src]["type"], data["type"], graph.nodes[dst]["type"])))
        return indices
