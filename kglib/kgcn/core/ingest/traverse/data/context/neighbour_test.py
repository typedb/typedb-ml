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
import unittest
from pathlib import Path

import grakn.client

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour
from kglib.kgcn.test.base import GraknServer

TEST_KEYSPACE = "test_schema"


class BaseGraknIntegrationTest:
    class GraknIntegrationTest(unittest.TestCase):

        session = None
        keyspace = TEST_KEYSPACE

        @classmethod
        def setUpClass(cls):
            client = grakn.client.GraknClient(uri="localhost:48555")
            cls.session = client.session(keyspace=cls.keyspace)

        @classmethod
        def tearDownClass(cls):
            cls.session.close()

        def setUp(self):
            self._tx = self.session.transaction().write()
            self._neighbour_finder = neighbour.NeighbourFinder()


class BaseTestNeighbourFinder:
    class TestNeighbourFinder(BaseGraknIntegrationTest.GraknIntegrationTest):
        
        def setUp(self):
            super(BaseTestNeighbourFinder.TestNeighbourFinder, self).setUp()
            self._grakn_thing = list(self._tx.query(self.query))[0].get(self.var)

        def test_role_is_in_neighbour_roles(self):
            for role in self.roles:
                self.assertIn(role, [r.role_label for r in self._res])

        def test_role_sets_equal(self):
            self.assertSetEqual(set(self.roles), {r.role_label for r in self._res})

        def test_neighbour_type_in_found_neighbours(self):
            self.assertIn(self.neighbour_type, [r.neighbour_thing.type_label for r in self._res])

        def test_num_results(self):
            self.assertEqual(self.num_results, len(self._res))


class TestNeighbourFinderFromEntity(BaseTestNeighbourFinder.TestNeighbourFinder):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    roles = ['employer', 'property', 'has']
    num_results = 3
    neighbour_type = 'employment'

    def setUp(self):
        super(TestNeighbourFinderFromEntity, self).setUp()
        self._res = list(self._neighbour_finder.find(self._grakn_thing.id, self._tx))


class TestNeighbourFinderFromRelation(BaseTestNeighbourFinder.TestNeighbourFinder):

    query = "match $x isa employment; get;"
    var = 'x'
    roles = ['employer', 'employee', 'has']
    num_results = 3
    neighbour_type = 'person'

    def setUp(self):
        super(TestNeighbourFinderFromRelation, self).setUp()
        self._res = list(self._neighbour_finder.find(self._grakn_thing.id, self._tx))


class TestNeighbourFinderFromAttribute(BaseTestNeighbourFinder.TestNeighbourFinder):

    query = "match $x isa job-title; get;"
    var = 'x'
    roles = ['has']
    num_results = 2
    neighbour_type = 'employment'

    def setUp(self):
        super(TestNeighbourFinderFromAttribute, self).setUp()
        self._res = list(self._neighbour_finder.find(self._grakn_thing.id, self._tx))


# class IntegrationTestNeighbourFinderFromDateAttribute(BaseTestNeighbourFinder.TestNeighbourFinder):
#     # Replicates the same issue as TestNeighbourFinderFromDateAttribute but using real animaltrede dataset
#     query = "match $attribute isa exchange-date 2016-01-01T00:00:00; limit 1; get;"
#     var = 'attribute'
#     roles = ['has']
#     num_results = 2
#     neighbour_type = 'import'
#     keyspace = 'animaltrade_train'
#
#     def setUp(self):
#         super(IntegrationTestNeighbourFinderFromDateAttribute, self).setUp()
#         self._res = list(itertools.islice(self._neighbour_finder.find(neighbour.NEIGHBOUR_PLAYS, self._grakn_thing.id, self._tx), 2))


class TestNeighbourFinderFromDateAttribute(BaseTestNeighbourFinder.TestNeighbourFinder):

    query = "match $attribute isa date-started; $attribute 2015-11-12T00:00; get;"
    var = 'attribute'
    roles = ['has']
    num_results = 1
    neighbour_type = 'project'

    def setUp(self):
        super(TestNeighbourFinderFromDateAttribute, self).setUp()
        self._res = list(self._neighbour_finder.find(self._grakn_thing.id, self._tx))


class TestFindLowestRoleFromRoleSups(BaseGraknIntegrationTest.GraknIntegrationTest):
    relation_query = "match $employment(employee: $roleplayer) isa employment; get;"
    role_query = "match $employment id {}; $person id {}; $employment($role: $person); get $role;"
    relation_var = 'employment'
    thing_var = 'roleplayer'
    role_var = 'role'

    def setUp(self):
        super(TestFindLowestRoleFromRoleSups, self).setUp()
        ans = list(self._tx.query(self.relation_query))[0]
        self._thing = ans.get(self.thing_var)
        self._relation = ans.get(self.relation_var)
        role_query = self.role_query.format(self._relation.id, self._thing.id)
        self._role_sups = [r.get(self.role_var) for r in self._tx.query(role_query)]

    def test_role_matches(self):
        role_found = neighbour.find_lowest_role_from_role_sups(self._role_sups)
        self.assertEqual('employee', role_found.label())

    def test_reversed_matches(self):
        role_found = neighbour.find_lowest_role_from_role_sups(list(reversed(self._role_sups)))
        self.assertEqual('employee', role_found.label())


class BaseTestBuildThing:
    class TestBuildThing(BaseGraknIntegrationTest.GraknIntegrationTest):
        def setUp(self):
            super(BaseTestBuildThing.TestBuildThing, self).setUp()

            self._grakn_thing = list(self._tx.query(self.query))[0].get(self.var)

            self._thing = neighbour.build_thing(self._grakn_thing)

        def test_id(self):
            self.assertEqual(self._thing.id, self._grakn_thing.id)

        def test_type_label(self):
            self.assertEqual(self._thing.type_label, self.type_label)

        def test_base_type_label(self):
            self.assertEqual(self._thing.base_type_label, self.base_type)


class TestBuildThingForEntity(BaseTestBuildThing.TestBuildThing):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    type_label = 'company'
    base_type = 'entity'


class TestBuildThingForRelation(BaseTestBuildThing.TestBuildThing):

    query = "match $x isa employment; get;"
    var = 'x'
    type_label = 'employment'
    base_type = 'relation'


class TestBuildThingForImplicitRelation(BaseTestBuildThing.TestBuildThing):

    query = "match $x isa @has-job-title; get;"
    var = 'x'
    type_label = '@has-job-title'
    base_type = 'relation'  # TODO do we want to see @has-attribute here?


class BaseTestBuildThingForAttribute:
    class TestBuildThingForAttribute(BaseTestBuildThing.TestBuildThing):

        def test_data_type(self):
            self.assertEqual(self._thing.data_type, self.data_type)

        def test_value(self):
            self.assertEqual(self._thing.value, self.value)


class TestBuildThingForStringAttribute(BaseTestBuildThingForAttribute.TestBuildThingForAttribute):

    query = "match $x isa job-title; get;"
    var = 'x'
    type_label = 'job-title'
    base_type = 'attribute'
    data_type = 'string'
    value = 'CEO'


if __name__ == "__main__":

    with GraknServer():
        with grakn.client.GraknClient(uri="localhost:48555") as client:
            with client.session(keyspace=TEST_KEYSPACE) as session:
                with session.transaction().write() as tx:
                    contents = Path(os.getenv("TEST_SRCDIR") + '/kglib/kglib/kgcn/test_data/schema.gql').read_text()
                    # tx.query(contents)
                    # tx.commit()

    # unittest.main()
