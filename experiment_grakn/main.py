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

from kglib.graph.create.from_queries import build_graph_from_queries
from experiment_grakn.data import get_examples

from grakn.client import GraknClient


def main():

    examples = get_examples()

    graphs = []

    with GraknClient(uri="localhost:48555") as client:
        with client.session(keyspace="genealogy") as session:

            for query_sampler_variable_graph_tuples in examples:

                with session.transaction().write() as tx:

                    combined_graph = build_graph_from_queries(query_sampler_variable_graph_tuples, tx)
                    graphs.append(combined_graph)


if __name__ == "__main__":
    main()
