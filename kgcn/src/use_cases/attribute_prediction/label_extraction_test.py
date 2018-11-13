import collections
import unittest
import unittest.mock as mock

import grakn
import grakn.service.Session.Concept.Concept as concept
import grakn.service.Session.util.ResponseReader as response

import kgcn.src.use_cases.attribute_prediction.label_extraction as label_extraction


class TestConceptLabelExtractor(unittest.TestCase):
    """
    Takes a query and performs it upon a Grakn keyspace in order to retrieve a set/variable number of concepts. These
    concepts should all have a label in the form of an attribute that they own.

    It should be possible to provide a method of sampling the concepts found. This is better performed downstream

    Labels could be of many formats. They could be categorical (long, date, boolean, string), or continuous (long,
    double). Providing this interpretation is important so that the desired output format is known.

    Multiple label prediction
    Often we want to predict multiple labels for an example. Therefore we may wish to provide a query for more than
    one attribute for each of our example concepts.

    Therefore, for each attribute being searched for, we need to provide the type of label: categorical or
    continuous, and if categorical, a list of the possible categories. This list could be built dynamically,
    but this may add too much complexity.

    The idea here will be that this extractor should be consistent, but for multiple keyspaces. In this way training
    and evaluation data can be kept separately but comparable datasets extracted from each. For this purpose each time
    the extractor is called it sjould take a transaction, which can only access a single keyspace.

    At this time this component is scoped to only work for the case where an owning concept owns one of each of the
    attributes being searched for.
    """

    def setUp(self):
        self._vars_config = ('x', collections.OrderedDict([('age_var', []),  # Continuous
                                                           ('gender_var', ['male', 'female'])  # Categorical
                                                           ])
                             )

        self._query = 'match $x isa person, has age $age_var, has gender $gender_var; limit 500; get;'

        self._person_mock = mock.Mock(concept.Entity)
        self._mock_age = mock.Mock(concept.Attribute)

        self._mock_age.value.return_value = 66
        self._mock_gender = mock.Mock(concept.Attribute)
        self._mock_gender.value.return_value = 'female'

        responses_for_variables = {'age_var': self._mock_age, 'gender_var': self._mock_gender, 'x': self._person_mock}

        def get(variable):
            return responses_for_variables[variable]

        self._grakn_tx = mock.Mock(grakn.Transaction)
        self._answer_mock = mock.Mock(response.Answer)
        self._answer_mock.get = mock.Mock()
        self._answer_mock.get.side_effect = get
        self._answers_iter_mock = iter([self._answer_mock])

        self._grakn_tx.query.return_value = self._answers_iter_mock

    def test_output_format_as_expected(self):

        concept_label_extractor = label_extraction.ConceptLabelExtractor(self._query, self._vars_config)
        concepts_with_labels = concept_label_extractor(self._grakn_tx, limit=None)

        expected_output = [(self._person_mock, {'age_var': [66], 'gender_var': [0, 1]})]
        self.assertListEqual(expected_output, concepts_with_labels)

    def test_get_called_for_each_attribute_variable(self):
        concept_label_extractor = label_extraction.ConceptLabelExtractor(self._query, self._vars_config)
        concept_label_extractor(self._grakn_tx, limit=None)
        self._answer_mock.get.assert_has_calls([mock.call('x'), mock.call('age_var'), mock.call('gender_var')])

    def test_value_called_for_each_attribute(self):
        concept_label_extractor = label_extraction.ConceptLabelExtractor(self._query, self._vars_config)
        concept_label_extractor(self._grakn_tx, limit=None)
        self._mock_age.value.assert_called_once()
        self._mock_gender.value.assert_called_once()
