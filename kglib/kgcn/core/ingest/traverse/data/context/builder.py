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

    def build_batch(self, session: grakn.client.Session, grakn_things: typ.List[neighbour.Thing]):
        things = [neighbour.build_thing(grakn_thing) for grakn_thing in grakn_things]

        thing_contexts = []
        for thing in things:
            tx = session.transaction().write()
            print(f'Opening transaction {tx}')
            thing_context = self.build(tx, thing)
            thing_contexts.append(thing_context)
            print(f'closing transaction {tx}')
            tx.close()
        context_batch = convert_thing_contexts_to_neighbours(thing_contexts)

        return context_batch

    def build(self, tx: grakn.client.Transaction, example_thing: neighbour.Thing):
        depth = 0
        return self._traverse_from_thing(example_thing, depth, (), tx)

    def _traverse_from_thing(self, starting_thing: neighbour.Thing, depth: int, indices: tuple, tx):

        nodes = {}
        if depth == 0:
            nodes[0] = [Node(indices, starting_thing)]

        if depth == self._max_depth:
            # This marks the end of the recursion, so there are no neighbours in the neighbourhood
            return nodes

        sampler = self._depth_samplers[depth]

        # Any concept could play a role in a relation if the schema permits it
        # Distinguish the concepts found as roles-played
        connections = self._neighbour_finder.find(starting_thing.id, tx)



        next_depth = depth + 1

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


# Could be renamed to a frame/situation/region/ROI(Region of Interest)/locale/zone
class ThingContext(utils.PropertyComparable):
    def __init__(self, thing: neighbour.Thing, neighbourhood: typ.List['Neighbour']):
        self.thing = thing
        self.neighbourhood = neighbourhood  # An iterator of `Neighbour`s


class Neighbour(utils.PropertyComparable):
    def __init__(self, role_label: (str, None), role_direction: (int, None), context: ThingContext):
        self.role_label = role_label
        self.role_direction = role_direction
        self.context = context


def convert_thing_contexts_to_neighbours(thing_contexts):
    """Dummy Neighbours so that a consistent data structure can be used right from the top level"""
    top_level_neighbours = [Neighbour(None, None, thing_context) for thing_context in thing_contexts]
    return top_level_neighbours
