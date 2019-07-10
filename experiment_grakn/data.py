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

import networkx as nx


def get_examples():
    return [get_example_graph_query_sampler_query_graph_tuples(example_id) for example_id in [0]]


def get_example_graph_query_sampler_query_graph_tuples(example_id):
    parentship_query = (
        f'match '
        f'$f isa family, has example-id {example_id}; '
        f'$p1 isa person; ($p1, $f);'
        f'$p2 isa person; ($p2, $f);'
        f'$r(parent: $p1, child: $p2) isa parentship;'
        f'get $p1, $p2, $r;'
    )
    print(parentship_query)

    g = nx.MultiDiGraph()
    g.add_node('p1')
    g.add_node('p2')
    g.add_node('r')
    g.add_edge('r', 'p1', type='parent')
    g.add_edge('r', 'p2', type='child')
    parentship_query_graph = g

    siblingship_query = (
        f'match '
        f'$f isa family, has example-id {example_id}; '
        f'$p1 isa person; ($p1, $f);'
        f'$r(sibling: $p1) isa siblingship;'
        f'get $p1, $r;'
    )
    print(siblingship_query)

    g = nx.MultiDiGraph()
    g.add_node('p1')
    g.add_node('p2')
    g.add_node('r')
    g.add_edge('r', 'p1', type='sibling')
    g.add_edge('r', 'p2', type='sibling')
    siblingship_query_graph = g

    candidate_siblingship_query = (
        f'match '
        f'$f isa family, has example-id {example_id}; '
        f'$p1 isa person; ($p1, $f);'
        f'$r(sibling: $p1) isa candidate-siblingship;'
        f'get $p1, $r;'
    )
    print(candidate_siblingship_query)

    candidate_siblingship_query_graph = siblingship_query_graph

    query_sampler_query_graph_tuples = [
        (parentship_query, lambda x: x, parentship_query_graph),
        (siblingship_query, lambda x: x, siblingship_query_graph),
        (candidate_siblingship_query, lambda x: x, candidate_siblingship_query_graph)
    ]

    return query_sampler_query_graph_tuples


if __name__ == "__main__":
    get_example_graph_query_sampler_query_graph_tuples(0)
