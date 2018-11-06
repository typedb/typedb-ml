import unittest

import grakn

import kgcn.src.neighbourhood.data.executor as ex


class BaseGraknIntegrationTests:
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
            self._executor = ex.TraversalExecutor(self._tx)
            self._concept = list(self._tx.query(self.query))[0].get(self.var)

        def test_role_is_in_neighbour_roles(self):
            self.assertIn(self.role, [r['role_label'] for r in self._res])

        def test_num_results(self):
            self.assertEqual(len(self._res), self.num_results)


class TestTraversalExecutorFromEntity(BaseGraknIntegrationTests.GraknIntegrationTest):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    role = 'employer'
    num_results = 3

    def setUp(self):
        super(TestTraversalExecutorFromEntity, self).setUp()
        self._res = list(self._executor(ex.TARGET_PLAYS, self._concept.id))


class TestTraversalExecutorFromRelationship(BaseGraknIntegrationTests.GraknIntegrationTest):

    query = "match $x isa employment; get;"
    var = 'x'
    role = 'employer'
    num_results = 2

    def setUp(self):
        super(TestTraversalExecutorFromRelationship, self).setUp()
        self._res = list(self._executor(ex.NEIGHBOUR_PLAYS, self._concept.id))


class BaseTestBuildConceptInfo:
    class TestBuildConceptInfo(BaseGraknIntegrationTests.GraknIntegrationTest):
        def setUp(self):
            super(BaseTestBuildConceptInfo.TestBuildConceptInfo, self).setUp()

            self._concept = list(self._tx.query(self.query))[0].get(self.var)

            self._concept_info = ex.build_concept_info(self._concept)

        def test_id(self):
            self.assertEqual(self._concept_info.id, self._concept.id)

        def test_type_label(self):
            self.assertEqual(self._concept_info.type_label, self.type_label)

        def test_base_type_label(self):
            self.assertEqual(self._concept_info.base_type_label, self.base_type)


class TestBuildConceptInfoForEntity(BaseTestBuildConceptInfo.TestBuildConceptInfo):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    type_label = 'company'
    base_type = 'entity'


class TestBuildConceptInfoForRelationship(BaseTestBuildConceptInfo.TestBuildConceptInfo):

    query = "match $x isa employment; get;"
    var = 'x'
    type_label = 'employment'
    base_type = 'relationship'


class TestBuildConceptInfoForImplicitRelationship(BaseTestBuildConceptInfo.TestBuildConceptInfo):

    query = "match $x isa @has-job-title; get;"
    var = 'x'
    type_label = '@has-job-title'
    base_type = 'relationship'  # TODO do we want to see @has-attribute here?


class BaseTestBuildConceptInfoForAttribute:
    class TestBuildConceptInfoForAttribute(BaseTestBuildConceptInfo.TestBuildConceptInfo):

        def test_data_type(self):
            self.assertEqual(self._concept_info.data_type, self.data_type)

        def test_value(self):
            self.assertEqual(self._concept_info.value, self.value)


class TestBuildConceptInfoForStringAttribute(BaseTestBuildConceptInfoForAttribute.TestBuildConceptInfoForAttribute):

    query = "match $x isa job-title; get;"
    var = 'x'
    type_label = 'job-title'
    base_type = 'attribute'
    data_type = 'string'
    value = 'CEO'
