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

import itertools


class Sampler:

    def __init__(self, sample_size, sampling_method, limit=None):
        """
        :param sample_size: the number of neighbours to sample at each depth
        :param sampling_method: method of sampling neighbours
        :param limit: limit the available population to this amount
        sampling speed but sacrifices randomness
        """
        self.sample_size = sample_size
        self._sampling_method = sampling_method
        self._limit = limit

    def __call__(self, population):

        # Rather than looking at all roleplayers and roles played, limit the number to a multiple of the number of
        # samples wanted. This makes the process pseudo-random, but saves a lot of time when querying
        if self._limit is not None:
            population = itertools.islice(population, self._limit)

        return self._sampling_method(population, self.sample_size)
