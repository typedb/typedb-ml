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

import numpy as np
import pandas as pd


class PMF:
    def __init__(self, variables, pmf_array, seed=None):
        """
        Probability Mass Function, the discrete equivalent of a Joint Probability Density Function

        Args:
            variables: An ordered dict, keys: the named variables of the function; values: the possible (discrete)
                values for each variable
            pmf_array: Probability Mass Function array: A numpy nd array of probabilities that corresponds with
                `variables`. Each variable is represented by a dimension of the array. For example, when there are 3
                variables, element (0, 0, 0) indicates the probability that all variables take the first value given
                for them in `variables`

        Raises:
            IndexError if `variables` and `pmf_array` are inconsistent
        """
        self._variables = variables
        self._pmf_array = pmf_array

        # Check that `self._variables` and `self._pmf_array` are consistent
        values_shape = tuple(
            len(discrete_values) for i, (variable, discrete_values) in enumerate(self._variables.items()))

        if values_shape != self._pmf_array.shape:
            raise IndexError(f'Variable values have combined shape {values_shape}, whereas the PMF array given has '
                             f'shape {self._pmf_array.shape}')

        if seed is not None:
            np.random.seed(seed)

    def select(self):
        """
        Select a set of variable values from the PMF, using the probabilities supplied in `pmf_array` as weights.

        Returns:
            A dict key: variable names; values: the chosen value of each variable
        """
        flattened = self._pmf_array.flatten()

        answer = {}

        indices = list(np.ndindex(self._pmf_array.shape))
        int_index = list(range(len(indices)))
        chosen_int = np.random.choice(int_index, p=flattened)
        chosen_index = indices[chosen_int]
        for index, (variable, discrete_values) in zip(chosen_index, self._variables.items()):
            answer[variable] = discrete_values[index]
        return answer

    def to_dataframe(self):
        """
        Creates a DataFrame of the PMF, most useful for visualisation purposes

        Returns:
            A pandas DataFrame, multi-indexed by the variables and their possible values

        """
        variables = list(self._variables.keys())
        variable_values = list(self._variables.values())
        index = pd.MultiIndex.from_product(variable_values, names=variables)

        return pd.DataFrame(self._pmf_array.flatten(), index=index)
