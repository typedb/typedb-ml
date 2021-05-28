#
#  Copyright (C) 2021 Vaticle
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
from grakn.client import Grakn, GraknSession, SessionType, GraknOptions, TransactionType
from kglib.utils.graph.thing.queries_to_networkx_graph import build_graph_from_queries


class GraknNetworkxDataSet:
    """
    Loading graphs based on queries from the Grakn database.
    Note: not dependent on PyTorch or Pytorch Geometric.
    """

    def __init__(
        self,
        example_indices: Sequence,
        get_query_handles_for_id: Callable,
        database: Optional[str] = None,
        uri: Optional[str] = "localhost:1729",
        session: Optional[GraknSession] = None,
        infer: bool = True,
        transform: Optional[Callable[[nx.Graph], nx.Graph]] = None,
    ):
        assert (database and uri) or session
        self._example_indices = example_indices
        self.get_query_handles_for_id = get_query_handles_for_id
        self._infer = infer
        self._transform = transform
        self._uri = uri
        self._database = database
        self._grakn_session = session

    @property
    def grakn_session(self):
        """
        Did this like this in an attempt to make it
        also work when using with a DataLoader with
        num_workers > 0.

        TODO: it does not, so look into this.
        """
        if not self._grakn_session:
            print("setting up session")
            print(self)
            client = TypeDB.core_client(self._uri)
            self._grakn_session = client.session(database=self._database, session_type=SessionType.DATA)
        return self._grakn_session

    def __len__(self):
        return len(self._example_indices)

    def __getitem__(self, idx):
        print(type(self._grakn_session))
        example_id = self._example_indices[idx]
        print(f"Fetching subgraph for example {example_id}")
        graph_query_handles = self.get_query_handles_for_id(example_id)

        options = GraknOptions.core()
        options.infer = self._infer

        with self.grakn_session.transaction(TransactionType.READ, options=options) as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx)
        graph.name = example_id
        if self._transform:
            graph = self._transform(graph)
        return graph