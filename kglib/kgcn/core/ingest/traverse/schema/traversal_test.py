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

import grakn.client

import kglib.kgcn.core.ingest.encode.encode as encode
import kglib.kgcn.core.ingest.traverse.schema.executor as ex
import kglib.kgcn.core.ingest.traverse.schema.traversal as trv
from kglib.utils.grakn.test.base import GraknServer

TEST_KEYSPACE = "test_schema"
TEST_URI = "localhost:48555"


class TestGetSchemaConceptTypes(unittest.TestCase):

    def setUp(self):
        client = grakn.client.GraknClient(uri=TEST_URI)
        session = client.session(keyspace=TEST_KEYSPACE)
        self._tx = session.transaction().write()

    def tearDown(self):
        self._tx.close()

    def _function_calls(self, query, include_implicit, include_metatypes):
        exec = ex.TraversalExecutor(self._tx)
        schema_concept_types = exec.get_schema_concept_types(query, include_implicit=include_implicit, include_metatypes=include_metatypes)
        labels = trv.labels_from_types(schema_concept_types)
        return labels

    def _filtering(self, query, num_types):
        with self.subTest('implicit_filtering'):
            labels = self._function_calls(query, False, True)
            # Make sure none of the type labels contain '@has-' which indicates that the type is implicit
            self.assertFalse(any(['@has-' in label for label in labels]))

        with self.subTest('metatype_filtering'):
            labels = self._function_calls(query, True, False)
            self.assertFalse(any([label in trv.METATYPE_LABELS for label in labels]))

        with self.subTest('all_members'):
            labels = list(self._function_calls(query, True, True))
            print(labels)

            with self.subTest("contains implicit"):
                self.assertTrue(any([label in trv.METATYPE_LABELS for label in labels]))

            with self.subTest("contains metatypes"):
                self.assertFalse(all([label in trv.METATYPE_LABELS for label in labels]))

            with self.subTest("length correct"):
                self.assertEqual(len(labels), num_types)

    def test_thing_filtering(self):
        self._filtering(encode.GET_THING_TYPES_QUERY, 19)

    def test_role_filtering(self):
        self._filtering(encode.GET_ROLE_TYPES_QUERY, 16)

    def test_integration(self):
        client = grakn.client.GraknClient(uri=TEST_URI)
        session = client.session(keyspace=TEST_KEYSPACE)
        tx = session.transaction().write()

        print("================= THINGS ======================")
        te = ex.TraversalExecutor(tx)
        schema_concept_types = te.get_schema_concept_types(encode.GET_THING_TYPES_QUERY, include_implicit=True,
                                                           include_metatypes=False)
        labels = trv.labels_from_types(schema_concept_types)
        print(list(labels))

        schema_concept_types = te.get_schema_concept_types(encode.GET_THING_TYPES_QUERY, include_implicit=True,
                                                           include_metatypes=False)
        super_types = trv.get_sups_labels_per_type(schema_concept_types, include_self=True, include_metatypes=False)
        print("==== super types ====")
        [print(type, super_types) for type, super_types in super_types.items()]

        print("================= ROLES ======================")
        schema_concept_types = te.get_schema_concept_types(encode.GET_ROLE_TYPES_QUERY, include_implicit=True,
                                                           include_metatypes=False)
        labels = trv.labels_from_types(schema_concept_types)
        print(list(labels))

        schema_concept_types = te.get_schema_concept_types(encode.GET_ROLE_TYPES_QUERY, include_implicit=True,
                                                           include_metatypes=False)
        super_types = trv.get_sups_labels_per_type(schema_concept_types, include_self=True, include_metatypes=False)
        print("==== super types ====")
        [print(type, super_types) for type, super_types in super_types.items()]


if __name__ == "__main__":

    with GraknServer() as gs:

        sp.check_call([
            'grakn', 'console', '-k', TEST_KEYSPACE, '-f',
            os.getenv("TEST_SRCDIR") + '/kglib/kglib/kgcn/test_data/schema.gql'
        ], cwd=gs.grakn_binary_location)

        unittest.main()
