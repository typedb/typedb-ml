
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

import kgcn.neighbourhood.data.sampling.ordered as ordered


def _gen(l):
    for i in l:
        yield i


class TestOrderedSample(unittest.TestCase):
    def setUp(self):
        self._p = list(range(5))

    def test_when_sample_size_less_than_population(self):

        population = _gen(self._p)
        n = 3
        first_n = ordered.ordered_sample(population, n)
        self.assertListEqual(list(first_n), self._p[:n])

    def test_when_sample_size_more_than_population(self):
        ns = [7, 10, 11, 15, 24]

        for n in ns:
            with self.subTest(f'population size = {len(self._p)}, n = {n}'):
                population = _gen(self._p)

                first_n = ordered.ordered_sample(population, n)

                self.assertListEqual(list(first_n), list(_gen(self._p))[:n])

    def test_when_sample_size_is_zero(self):

        population = _gen(self._p)
        n = 0
        first_n = ordered.ordered_sample(population, n)
        self.assertListEqual(list(first_n), [])

    def test_when_population_size_is_zero(self):

        population = _gen([])
        n = 5
        with self.assertRaises(ValueError) as context:
            first_n = ordered.ordered_sample(population, n)

        self.assertTrue('Population is empty' in str(context.exception))


if __name__ == "__main__":
    unittest.main()
