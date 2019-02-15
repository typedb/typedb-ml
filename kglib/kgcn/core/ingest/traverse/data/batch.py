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
import grakn

import kglib.kgcn.core.ingest.traverse.data.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.context as context


class BatchContextBuilder:

    def __init__(self, traversal_samplers):
        self._traversal_samplers = traversal_samplers

    def __call__(self, session, grakn_things):
        ################################################################################################################
        # Neighbour Traversals
        ################################################################################################################
        things = [neighbour.build_thing(grakn_thing) for grakn_thing in grakn_things]

        data_executor = neighbour.NeighbourFinder()
        context_builder = context.ContextBuilder(data_executor, self._traversal_samplers)

        thing_contexts = []
        for thing in things:
            tx = session.transaction(grakn.TxType.WRITE)
            print(f'Opening transaction {tx}')
            thing_context = context_builder(thing, tx)
            context.collect_to_tree(thing_context)
            thing_contexts.append(thing_context)
            print(f'closing transaction {tx}')
            tx.close()
        context_batch = convert_thing_contexts_to_neighbours(thing_contexts)

        return context_batch


def convert_thing_contexts_to_neighbours(thing_contexts):
    """Dummy Neighbours so that a consistent data structure can be used right from the top level"""
    top_level_neighbours = [context.Neighbour(None, None, thing_context) for thing_context in thing_contexts]
    return top_level_neighbours
