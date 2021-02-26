from grakn.client import GraknClient
from kglib.utils.graph.thing.queries_to_networkx_graph import build_graph_from_queries


class GraknNetworkxDataSet:
    """
    Loading graphs based on queries from the Grakn database.
    Note: not dependent on PyTorch or Pytorch Geometric.
    """

    def __init__(
        self,
        example_indices,
        get_query_handles_for_id,
        database,
        uri="localhost:1729",
        infer=True,
        transform=None,
    ):
        self._example_indices = example_indices
        self.get_query_handles_for_id = get_query_handles_for_id
        self._infer = infer
        self._transform = transform
        self._uri = uri
        self._database = database

        self._grakn_session = None

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
            client = GraknClient.core(address=self._uri)
            self._grakn_session = client.session(database=self._keyspace, session_type=SessionType.DATA)
        return self._grakn_session

    def __len__(self):
        return len(self._example_indices)

    def __getitem__(self, idx):
        example_id = self._example_indices[idx]
        print(f"Fetching subgraph for example {example_id}")
        graph_query_handles = self.get_query_handles_for_id(example_id)

        options = GraknOptions.core()
        options.infer = True;

        with self.grakn_session.transaction(TransactionType.READ, options=options) as tx:
            # Build a graph from the queries, samplers, and query graphs
            graph = build_graph_from_queries(graph_query_handles, tx)
        graph.name = example_id
        if self._transform:
            graph = self._transform(graph)
        return graph