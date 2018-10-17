import unittest
import grakn
import grakn_graphsage.src.neighbourhood.schema.traversal as trv


class TestGetSchemaConceptTypes(unittest.TestCase):

    def setUp(self):
        client = grakn.Grakn(uri="localhost:48555")
        session = client.session(keyspace="test_schema")
        self._tx = session.transaction(grakn.TxType.WRITE)

    def tearDown(self):
        self._tx.close()

    def _function_calls(self, query, include_implicit, include_metatypes):
        schema_concept_types = trv.get_schema_concept_types(self._tx, query, include_implicit=include_implicit, include_metatypes=include_metatypes)
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
        self._filtering(trv.GET_THING_TYPES_QUERY, 16)

    def test_role_filtering(self):
        self._filtering(trv.GET_ROLE_TYPES_QUERY, 14)