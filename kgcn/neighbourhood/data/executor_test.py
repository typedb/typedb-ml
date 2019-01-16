import itertools
import unittest

import grakn

import kgcn.neighbourhood.data.executor as ex


class BaseGraknIntegrationTest:
    class GraknIntegrationTest(unittest.TestCase):

        session = None
        keyspace = "test_schema"

        @classmethod
        def setUpClass(cls):
            client = grakn.Grakn(uri="localhost:48555")
            cls.session = client.session(keyspace=cls.keyspace)

        @classmethod
        def tearDownClass(cls):
            cls.session.close()

        def setUp(self):
            self._tx = self.session.transaction(grakn.TxType.WRITE)
            self._executor = ex.TraversalExecutor()


class BaseTestTraversalExecutor:
    class TestTraversalExecutor(BaseGraknIntegrationTest.GraknIntegrationTest):
        
        def setUp(self):
            super(BaseTestTraversalExecutor.TestTraversalExecutor, self).setUp()
            self._concept = list(self._tx.query(self.query))[0].get(self.var)

        def test_role_is_in_neighbour_roles(self):
            for role in self.roles:
                self.assertIn(role, [r['role_label'] for r in self._res])

        def test_role_sets_equal(self):
            self.assertSetEqual(set(self.roles), {r['role_label'] for r in self._res})

        def test_neighbour_type_in_found_neighbours(self):
            self.assertIn(self.neighbour_type, [r['neighbour_info'].type_label for r in self._res])

        def test_num_results(self):
            self.assertEqual(self.num_results, len(self._res))


class TestTraversalExecutorFromEntity(BaseTestTraversalExecutor.TestTraversalExecutor):

    query = "match $x isa company, has name 'Google'; get;"
    var = 'x'
    roles = ['employer', 'property', 'has']
    num_results = 3
    neighbour_type = 'employment'

    def setUp(self):
        super(TestTraversalExecutorFromEntity, self).setUp()
        self._res = list(self._executor(ex.TARGET_PLAYS, self._concept.id, self._tx))


class TestTraversalExecutorFromRelationship(BaseTestTraversalExecutor.TestTraversalExecutor):

    query = "match $x isa employment; get;"
    var = 'x'
    roles = ['employer', 'employee', 'has']
    num_results = 3
    neighbour_type = 'person'

    def setUp(self):
        super(TestTraversalExecutorFromRelationship, self).setUp()
        self._res = list(self._executor(ex.NEIGHBOUR_PLAYS, self._concept.id, self._tx))


class TestTraversalExecutorFromAttribute(BaseTestTraversalExecutor.TestTraversalExecutor):

    query = "match $x isa job-title; get;"
    var = 'x'
    roles = ['has']
    num_results = 2
    neighbour_type = 'employment'

    def setUp(self):
        super(TestTraversalExecutorFromAttribute, self).setUp()
        self._res = list(self._executor(ex.NEIGHBOUR_PLAYS, self._concept.id, self._tx))


# class IntegrationTestTraversalExecutorFromDateAttribute(BaseTestTraversalExecutor.TestTraversalExecutor):
#     # Replicates the same issue as TestTraversalExecutorFromDateAttribute but using real animaltrede dataset
#     query = "match $attribute isa exchange-date 2016-01-01T00:00:00; limit 1; get;"
#     var = 'attribute'
#     roles = ['has']
#     num_results = 2
#     neighbour_type = 'import'
#     keyspace = 'animaltrade_train'
#
#     def setUp(self):
#         super(IntegrationTestTraversalExecutorFromDateAttribute, self).setUp()
#         self._res = list(itertools.islice(self._executor(ex.NEIGHBOUR_PLAYS, self._concept.id, self._tx), 2))


class TestTraversalExecutorFromDateAttribute(BaseTestTraversalExecutor.TestTraversalExecutor):

    query = "match $attribute isa date-started 2015-11-12T00:00; limit 1; get;"
    var = 'attribute'
    roles = ['has']
    num_results = 1
    neighbour_type = 'project'

    def setUp(self):
        super(TestTraversalExecutorFromDateAttribute, self).setUp()
        self._res = list(self._executor(ex.NEIGHBOUR_PLAYS, self._concept.id, self._tx))


class TestFindLowestRoleFromRoleSups(BaseGraknIntegrationTest.GraknIntegrationTest):
    relationship_query = "match $employment(employee: $roleplayer) isa employment; get;"
    role_query = "match $employment id {}; $person id {}; $employment($role: $person); get $role;"
    relationship_var = 'employment'
    thing_var = 'roleplayer'
    role_var = 'role'

    def setUp(self):
        super(TestFindLowestRoleFromRoleSups, self).setUp()
        ans = list(self._tx.query(self.relationship_query))[0]
        self._thing = ans.get(self.thing_var)
        self._relationship = ans.get(self.relationship_var)
        role_query = self.role_query.format(self._relationship.id, self._thing.id)
        self._role_sups = [r.get(self.role_var) for r in self._tx.query(role_query)]

    def test_role_matches(self):
        role_found = ex.find_lowest_role_from_rols_sups(self._role_sups)
        self.assertEqual('employee', role_found.label())

    def test_reversed_matches(self):
        role_found = ex.find_lowest_role_from_rols_sups(list(reversed(self._role_sups)))
        self.assertEqual('employee', role_found.label())


class BaseTestBuildConceptInfo:
    class TestBuildConceptInfo(BaseGraknIntegrationTest.GraknIntegrationTest):
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


if __name__ == "__main__":
    unittest.main()
