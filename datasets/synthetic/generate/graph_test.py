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

import unittest

from  datasets.synthetic.generate.graph import ExampleGenerator


class MockPMF:
    def __init__(self):
        self._counter = 0

    def select(self):
        if self._counter == 0:
            self._counter += 1
            return {'Flu': False, 'Meningitis': True}
        else:
            return {'Flu': True, 'Meningitis': False}


class TestGenerateExample(unittest.TestCase):
    def test_example_generated_as_expected(self):
        def base_insertion_query_fn(variable_values, example_id):
            return [(f'insert '
                     f'$p isa person, '
                     f'has example-id '
                     f'{example_id};')]

        def meningitis_query_fn(variable_values, example_id):
            if variable_values['Meningitis']:
                return [(f'match'
                         f'$d isa meningitis;'
                         f'$p isa person, has example-id '
                         f'{example_id};'
                         f'insert'
                         f'(patient: $p, diagnosed-disease: $d) '
                         f'isa diagnosis;')]

        def flu_query_fn(variable_values, example_id):
            if variable_values['Flu']:
                return [(f'match'
                         f'$p isa person, has example-id '
                         f'{example_id};'
                         f'(patient: $p, diagnosed-disease: $d) isa diagnosis;'
                         f'$s isa flu;'
                         f'insert'
                         f'(presented-symptom: $s, symptomatic-patient: $p) isa '
                         f'symptom-presentation;')]

        """
        We need to pass in a mapping from the values that are picked from the PMF and what queries to execute,
        also providing any unique identifiers needed so that subsequent queries can operate on the same example
        subgraph.
        This permits changing attribute values in queries according to the variable_values, and/or making multiple
        insertions of the same kind. Something of the form:
        
        def feature_func(variable_values, example_id):
            queries_to_execute = []
            return queries_to_execute
        """

        pmf = MockPMF()

        query_feature_fns = [base_insertion_query_fn,
                             meningitis_query_fn,
                             flu_query_fn]

        # TODO ExampleGenerator need to track the example_ids that have been used already, or query for the max,
        #  or only be used in batch mode
        gen = ExampleGenerator(query_feature_fns, pmf)
        examples = [gen.generate_example() for _ in range(2)]

        expected_examples = [
            [
                (f'insert '
                 f'$p isa person, '
                 f'has example-id 0;'),
                (f'match'
                 f'$d isa meningitis;'
                 f'$p isa person, has example-id 0;'
                 f'insert'
                 f'(patient: $p, diagnosed-disease: $d) '
                 f'isa diagnosis;')
            ],
            [
                (f'insert '
                 f'$p isa person, '
                 f'has example-id 1;'),
                (f'match'
                 f'$p isa person, has example-id 1;'
                 f'(patient: $p, diagnosed-disease: $d) isa diagnosis;'
                 f'$s isa flu;'
                 f'insert'
                 f'(presented-symptom: $s, symptomatic-patient: $p) isa '
                 f'symptom-presentation;')
            ]
        ]

        self.assertListEqual(expected_examples, examples)


if __name__ == '__main__':
    unittest.main()
