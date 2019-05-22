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

import typing as typ

import grakn.client

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.context.utils as utils


def update_dict_lists(dict_to_add, dict_to_update):
    for key, contained_list in dict_to_add.items():
        dict_to_update.setdefault(key, [])[0:0] = contained_list
    return dict_to_update


class ContextBuilder:
    def __init__(self, depth_samplers, neighbour_finder=neighbour.NeighbourFinder()):
        self._neighbour_finder = neighbour_finder
        self._depth_samplers = depth_samplers
        self._max_depth = len(self._depth_samplers)

    def build_batch(self, session: grakn.client.Session, example_things: typ.Iterator[neighbour.Thing]):

        thing_contexts = []
        for thing in example_things:
            tx = session.transaction().write()
            print(f'Opening transaction {tx}')
            thing_context = self.build(tx, thing)
            thing_contexts.append(thing_context)
            print(f'closing transaction {tx}')
            tx.close()

        return thing_contexts

    def build(self, tx: grakn.client.Transaction, example_thing: neighbour.Thing):
        depth = self._max_depth
        return self._traverse_from_thing(example_thing, depth, (), tx)

    def _traverse_from_thing(self, starting_thing: neighbour.Thing, depth: int, indices: tuple, tx):

        nodes = {}
        if depth == self._max_depth:
            nodes[self._max_depth] = [Node(indices, starting_thing)]

        if depth == 0:
            # This marks the end of the recursion, so there are no neighbours in the neighbourhood
            return nodes

        sampler = self._depth_samplers[self._max_depth - depth]

        # Any concept could play a role in a relation if the schema permits it
        # Distinguish the concepts found as roles-played
        connections = self._neighbour_finder.find(starting_thing.id, tx)

        next_depth = depth - 1

        # Sample the neighbourhood and iterate over the results
        for i, connection in enumerate(sampler(connections)):
            next_indices = (i,) + indices
            nodes.setdefault(next_depth, []).append(Node(indices=next_indices, thing=connection.neighbour_thing,
                                                         role_label=connection.role_label,
                                                         role_direction=connection.role_direction))
            child_nodes = self._traverse_from_thing(connection.neighbour_thing, next_depth, next_indices, tx)
            nodes = update_dict_lists(nodes, child_nodes)

        return nodes


class Node(utils.PropertyComparable):
    def __init__(self, indices, thing, role_label=None, role_direction=None):
        self.indices = indices
        self.role_label = role_label
        self.role_direction = role_direction
        self.thing = thing
