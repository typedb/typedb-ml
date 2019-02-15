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

import collections
import itertools
import typing as typ

import grakn

import kglib.kgcn.core.ingest.traverse.data.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.utils as utils


class ContextBuilder:
    def __init__(self, depth_samplers, neighbour_finder=neighbour.NeighbourFinder()):
        self._neighbour_finder = neighbour_finder
        self._depth_samplers = depth_samplers

    def build_batch(self, session: grakn.Session, grakn_things: typ.List[neighbour.Thing]):
        things = [neighbour.build_thing(grakn_thing) for grakn_thing in grakn_things]

        thing_contexts = []
        for thing in things:
            tx = session.transaction(grakn.TxType.WRITE)
            print(f'Opening transaction {tx}')
            thing_context = self.build(tx, thing)
            collect_to_tree(thing_context)
            thing_contexts.append(thing_context)
            print(f'closing transaction {tx}')
            tx.close()
        context_batch = convert_thing_contexts_to_neighbours(thing_contexts)

        return context_batch

    def build(self, tx: grakn.Transaction, example_thing: neighbour.Thing):
        depth = len(self._depth_samplers)
        return self._traverse(example_thing, depth, tx)

    def _traverse(self, starting_thing_context: neighbour.Thing, depth: int, tx):

        if depth == 0:
            # This marks the end of the recursion, so there are no neighbours in the neighbourhood
            return ThingContext(thing=starting_thing_context, neighbourhood=[])

        sampler = self._depth_samplers[-depth]
        next_depth = depth - 1
        # next_neighbour_samplers = self._depth_samplers[1:]

        # Different cases for traversal

        # Any concept could play a role in a relationship if the schema permits it

        # Distinguish the concepts found as roles-played
        roles_played = self._neighbour_finder(neighbour.ROLES_PLAYED, starting_thing_context.id, tx)

        neighbourhood = self._get_neighbour(roles_played, next_depth, tx)

        thing_context = ThingContext(thing=starting_thing_context, neighbourhood=neighbourhood)

        # TODO If user doesn't attach anything to impicit @has relationships, then these could be filtered out. Instead
        # another query would be required: "match $x id {}, has attribute $attribute; get $attribute;"
        # We would use TARGET_PLAYS with Role "has" or role "@has-< attribute type name >"

        # if target_thing.metatype_label == 'entity':
        #     # Nothing special to do in this case?
        #     pass
        # elif target_thing.metatype_label == 'attribute':
        #     # Do anything specific to attribute values
        #     # Optionally stop further propagation through attributes, since they are shared across the knowledge
        #     # graph so this may not provide relevant information

        if starting_thing_context.base_type_label == 'relationship':
            # Find its roleplayers
            roleplayers = self._neighbour_finder(neighbour.ROLEPLAYERS, starting_thing_context.id, tx)

            # Chain the iterators together, so that after getting the roles played you get the roleplayers
            thing_context.neighbourhood = itertools.chain(
                thing_context.neighbourhood,
                self._get_neighbour(roleplayers, next_depth, tx))

        # Randomly sample the neighbourhood
        thing_context.neighbourhood = sampler(thing_context.neighbourhood)

        return thing_context

    def _get_neighbour(self, role_and_concept_info_iterator, depth, tx):

        for connection in role_and_concept_info_iterator:
            neighbour_context = self._traverse(connection['neighbour_thing'], depth, tx)

            yield Neighbour(role_label=connection['role_label'], role_direction=connection['role_direction'],
                            context=neighbour_context)


# Could be renamed to a frame/situation/region/ROI(Region of Interest)/locale/zone
class ThingContext(utils.PropertyComparable):
    def __init__(self, thing: neighbour.Thing, neighbourhood: collections.Generator):
        self.thing = thing
        self.neighbourhood = neighbourhood  # An iterator of `Neighbour`s


class Neighbour(utils.PropertyComparable):
    def __init__(self, role_label: (str, None), role_direction: (int, None), context: ThingContext):
        self.role_label = role_label
        self.role_direction = role_direction
        self.context = context


def collect_to_tree(thing_context):
    """
    Given the neighbour generators, yield the fully populated tree of each of the target concept's neighbours
    :param thing_context:
    :return:
    """
    if thing_context is not None:
        thing_context.neighbourhood = materialise_subordinate_neighbours(thing_context)
        for neighbour in thing_context.neighbourhood:
            collect_to_tree(neighbour.context)

    return thing_context


def materialise_subordinate_neighbours(thing_context):
    """
    Build the list of all of the neighbours immediately "beneath" this concept. By beneath, meaning belonging to one
    layer deeper in the neighbour graph
    :param thing_context:
    :return:
    """
    return [neighbour for neighbour in thing_context.neighbourhood]


def flatten_tree(neighbours):
    all_connections = []

    for neighbour in neighbours:
        ci = neighbour.context.thing
        all_connections.append(
            (neighbour.role_label,
             neighbour.role_direction,
             ci.type_label,
             ci.base_type_label,
             ci.id,
             ci.data_type,
             ci.value
             ))

        all_connections += flatten_tree(neighbour.context.neighbourhood)  # List of neighbour roles
    return all_connections


def get_max_depth(thing_context: ThingContext):
    """
    Find the length of the deepest aggregation path
    :param thing_context:
    :return:
    """

    if len(thing_context.neighbourhood) == 0:
        return 0
    else:
        max_depth = 0
        for neighbour in thing_context.neighbourhood:
            m = get_max_depth(neighbour.context)
            if m > max_depth:
                max_depth = m
        return max_depth + 1


def convert_thing_contexts_to_neighbours(thing_contexts):
    """Dummy Neighbours so that a consistent data structure can be used right from the top level"""
    top_level_neighbours = [Neighbour(None, None, thing_context) for thing_context in thing_contexts]
    return top_level_neighbours