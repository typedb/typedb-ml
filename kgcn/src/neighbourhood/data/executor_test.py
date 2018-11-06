
import unittest

import grakn

import kgcn.src.neighbourhood.data.executor as ex


class GraknIntegrationTest(unittest.TestCase):

    session = None

    @classmethod
    def setUpClass(cls):
        client = grakn.Grakn(uri="localhost:48555")
        cls.session = client.session(keyspace="test_schema")

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

    def setUp(self):
        self._tx = self.session.transaction(grakn.TxType.WRITE)


class TestTraversalExecutor(GraknIntegrationTest):

    def setUp(self):
        # self._tx = self.session.transaction(grakn.TxType.WRITE)
        super(TestTraversalExecutor, self).setUp()
        self._executor = ex.TraversalExecutor(self._tx)

    def test_entity(self):

        entity_query = "match $x isa company, has name 'Google'; get;"

        concept = list(self._tx.query(entity_query))[0].get('x')
        res = self._executor(ex.TARGET_PLAYS, concept.id)


class TestBuildConceptInfo(GraknIntegrationTest):
    def setUp(self):
        super(TestBuildConceptInfo, self).setUp()

        self._concept = list(self._tx.query(self.query))[0].get(self.var)

        self._concept_info = ex.build_concept_info(self._concept)

    def test_id(self):
        self.assertEqual(self._concept_info.id, self._concept.id)

    def test_type_label(self):
        self.assertEqual(self._concept_info.type_label, self.type_label)

    def test_base_type_label(self):
        self.assertEqual(self._concept_info.base_type_label, self.base_type)


class TestBuildConceptInfoForEntity(TestBuildConceptInfo):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    type_label = 'company'
    base_type = 'entity'


class TestBuildConceptInfoForRelationship(TestBuildConceptInfo):

    query = "match $x isa employment; get;"
    var = 'x'
    type_label = 'employment'
    base_type = 'relationship'


class TestBuildConceptInfoForAttribute(TestBuildConceptInfo):

    def test_data_type(self):
        self.assertEqual(self._concept_info.data_type, self.data_type)

    def test_value(self):
        self.assertEqual(self._concept_info.value, self.value)


class TestBuildConceptInfoForStringAttribute(TestBuildConceptInfoForAttribute):

    query = "match $x isa job-title; get;"
    var = 'x'
    type_label = 'job-title'
    base_type = 'attribute'
    data_type = 'string'
    value = 'CEO'
