#
#  Copyright (C) 2022 Vaticle
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

import numpy as np
import pandas as pd

from pmf import PMF


class TestPMF(unittest.TestCase):

    def test_high_probability_value_selected_as_expected(self):

        a = np.zeros([2, 2, 2, 2], dtype=np.float)
        a[0, 1, 1, 1] = 1.0

        pmf = PMF({'Flu': [False, True], 'Meningitis': [False, True], 'Light Sensitivity': [False, True],
                   'Fever': [False, True]}, a)
        choice = pmf.select()
        expected_choice = {'Flu': False, 'Meningitis': True, 'Light Sensitivity': True, 'Fever': True}
        self.assertDictEqual(expected_choice, choice)

    def test_exception_raised_if_probabilities_and_values_inconsistent(self):

        a = np.zeros([2, 2, 2, 1], dtype=np.float)

        with self.assertRaises(IndexError) as context:
            PMF({'Flu': [False, True], 'Meningitis': [False, True], 'Light Sensitivity': [False, True],
                 'Fever': [False, True]}, a)

        self.assertEqual(str(context.exception), ('Variable values have combined shape (2, 2, 2, 2), whereas the PMF '
                                                  'array given has shape (2, 2, 2, 1)'))

    def test_to_dataframe_is_as_expected(self):
        features = ['Flu', 'Meningitis', 'Light Sensitivity', 'Fever']
        feat_values = [[False, True], [False, True], [False, True], [False, True]]
        index = pd.MultiIndex.from_product(feat_values, names=features)

        a = np.zeros([2, 2, 2, 2])
        a[0, 1, 1, 1] = 1.0

        pmf = PMF({'Flu': [False, True], 'Meningitis': [False, True], 'Light Sensitivity': [False, True],
                   'Fever': [False, True]}, a)

        df = pmf.to_dataframe()

        expected_df = pd.DataFrame(a.flatten(), index=index)
        pd.testing.assert_frame_equal(expected_df, df)


if __name__ == '__main__':
    unittest.main()
