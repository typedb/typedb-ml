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
import os
import subprocess as sp
import unittest
import unittest.mock as mock

import grakn.client

import kglib.kgcn.core.ingest.traverse.data.context.builder as builder
import kglib.kgcn.core.ingest.traverse.data.context.builder_mocks as mocks
import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
import kglib.kgcn.core.ingest.traverse.data.sample.ordered as ordered
import kglib.kgcn.core.ingest.traverse.data.sample.sample as samp
from kglib.kgcn.test.base import GraknServer

TEST_KEYSPACE = "test_schema"
TEST_URI = "localhost:48555"


class ITContextBuilder(unittest.TestCase):

    def test_build_context_for_0_hop(self):
        starting_thing = neighbour.Thing("0", "person", "entity")

        samplers = []
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build(mock.Mock(grakn.client.Transaction), starting_thing)
        expected_context = {
            0: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
        }
        self.assertEqual(expected_context, thing_context)

    def test_build_context_for_1_hop(self):

        starting_thing = neighbour.Thing("0", "person", "entity")

        samplers = [samp.Sampler(2, ordered.ordered_sample, limit=2)]
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build(mock.Mock(grakn.client.Transaction), starting_thing)

        expected_context = {
            0: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
                ],
            1: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
        }
        self.assertEqual(expected_context, thing_context)

    def test_build_context_for_2_hop(self):

        starting_thing = neighbour.Thing("0", "person", "entity")

        samplers = [samp.Sampler(2, ordered.ordered_sample, limit=2), samp.Sampler(2, ordered.ordered_sample, limit=2)]
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build(mock.Mock(grakn.client.Transaction), starting_thing)

        expected_context = {
            2: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
            1: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
                ],
            0: [builder.Node((0, 0), neighbour.Thing("0", "person", "entity"), "has", neighbour.TARGET_PLAYS),
                # Note that (0, 1) is reversed compared to the natural expectation
                builder.Node((0, 1), neighbour.Thing("3", "company", "entity"), "employer", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1, 1), neighbour.Thing("0", "person", "entity"), "employee", neighbour.NEIGHBOUR_PLAYS),
                ]
        }
        self.assertEqual(expected_context, thing_context)

    def test_build_context_for_3_hop(self):

        starting_thing = neighbour.Thing("0", "person", "entity")

        samplers = [samp.Sampler(2, ordered.ordered_sample, limit=2), samp.Sampler(2, ordered.ordered_sample, limit=2),
                    samp.Sampler(3, ordered.ordered_sample, limit=3)]
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build(mock.Mock(grakn.client.Transaction), starting_thing)

        expected_context = {
            3: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
            2: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1,), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
                ],
            1: [builder.Node((0, 0), neighbour.Thing("0", "person", "entity"), "has", neighbour.TARGET_PLAYS),
                # Note that (0, 1) is reversed compared to the natural expectation
                builder.Node((0, 1), neighbour.Thing("3", "company", "entity"), "employer", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1, 1), neighbour.Thing("0", "person", "entity"), "employee", neighbour.NEIGHBOUR_PLAYS),
                ],
            0: [builder.Node((0, 0, 0), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1, 0, 0), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
                builder.Node((0, 0, 1), neighbour.Thing("4", "name", "attribute", data_type='string', value='Google'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1, 0, 1), neighbour.Thing("4", "name", "attribute", data_type='string', value='Google'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((2, 0, 1), neighbour.Thing("4", "name", "attribute", data_type='string', value='Google'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((0, 1, 1), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                builder.Node((1, 1, 1), neighbour.Thing("2", "employment", "relation"), "employee", neighbour.TARGET_PLAYS),
            ]

        }
        self.assertEqual(expected_context, thing_context)

    def test_build_context_with_sampling_limit_for_1_hop__limits_context(self):

        starting_thing = neighbour.Thing("0", "person", "entity")

        samplers = [samp.Sampler(2, ordered.ordered_sample, limit=1)]
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build(mock.Mock(grakn.client.Transaction), starting_thing)

        expected_context = {
            1: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
            0: [builder.Node((0,), neighbour.Thing("1", "name", "attribute", data_type='string', value='Sundar Pichai'),
                             "has", neighbour.NEIGHBOUR_PLAYS),
                ]
        }
        self.assertEqual(expected_context, thing_context)

    def test_build_context_batch(self):
        batch = [neighbour.Thing("0", "person", "entity"), neighbour.Thing("0", "person", "entity")]

        samplers = []
        context_builder = builder.ContextBuilder(samplers, neighbour_finder=mocks.MockNeighbourFinder())

        thing_context = context_builder.build_batch(mock.Mock(grakn.client.Session), batch)
        expected_context_batch = [{
            0: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
        },
            {
                0: [builder.Node((), neighbour.Thing("0", "person", "entity"))],
            }
        ]
        self.assertEqual(expected_context_batch, thing_context)


class ITContextBuilderWithRealGrakn(unittest.TestCase):

    session = None

    @classmethod
    def setUpClass(cls):
        client = grakn.client.GraknClient(uri=TEST_URI)
        cls.session = client.session(keyspace=TEST_KEYSPACE)

        entity_query = "match $x isa company, has name 'Google'; get;"
        cls._tx = cls.session.transaction().write()

        neighbour_sample_sizes = (4, 3)

        sampling_method = ordered.ordered_sample

        samplers = []
        for sample_size in neighbour_sample_sizes:
            samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

        grakn_thing = next(cls._tx.query(entity_query)).get('x')

        thing = neighbour.build_thing(grakn_thing)

        context_builder = builder.ContextBuilder(samplers)

        cls.context = context_builder.build(cls._tx, thing)

    @classmethod
    def tearDownClass(cls):
        cls._tx.close()
        cls.session.close()

    def test_context_role_labels_not_absent(self):
        role_labels = []
        for node_list in [self.context[0], self.context[1]]:
            for node in node_list:
                role_labels.append(node.role_label)

        role_labels_absent = [label in ['', None] for label in role_labels]
        self.assertFalse(any(role_labels_absent))

    def test_context_type_labels_not_absent(self):
        type_labels = []
        for node_list in self.context.values():
            for node in node_list:
                type_labels.append(node.thing.type_label)

        type_labels_absent = [label in ['', None] for label in type_labels]
        self.assertFalse(any(type_labels_absent))

    def test_context_attribute_values_not_none(self):

        attribute_values = []
        for node_list in self.context.values():
            for node in node_list:
                if node.thing.base_type_label == 'attribute':
                    attribute_values.append(node.thing.value)

        attribute_values_absent = [label in ['', None] for label in attribute_values]

        self.assertFalse(any(attribute_values_absent))

    def test_context_entity_and_relation_values_are_none(self):

        thing_values = []
        for node_list in self.context.values():
            for node in node_list:
                if node.thing.base_type_label in ['entity', 'relation']:
                    thing_values.append(node.thing.value)

        thing_values_absent = [label in ['', None] for label in thing_values]

        self.assertTrue(all(thing_values_absent))

    def test_context_attribute_datatypes_not_none(self):
        attribute_datatypes = []
        for node_list in self.context.values():
            for node in node_list:
                if node.thing.base_type_label == 'attribute':
                    attribute_datatypes.append(node.thing.data_type)

        attribute_datatypes_absent = [label in ['', None] for label in attribute_datatypes]

        self.assertFalse(any(attribute_datatypes_absent))

    def test_context_entity_and_relation_datatypes_are_none(self):

        thing_datatypes = []
        for node_list in self.context.values():
            for node in node_list:
                if node.thing.base_type_label in ['entity', 'relation']:
                    thing_datatypes.append(node.thing.data_type)

        thing_datatypes_absent = [label in ['', None] for label in thing_datatypes]

        self.assertTrue(all(thing_datatypes_absent))


if __name__ == "__main__":

    with GraknServer() as gs:

        sp.check_call([
            'grakn', 'console', '-k', TEST_KEYSPACE, '-f',
            os.getenv("TEST_SRCDIR") + '/kglib/kglib/kgcn/test_data/schema.gql'
        ], cwd=gs.grakn_binary_location)

        unittest.main()
