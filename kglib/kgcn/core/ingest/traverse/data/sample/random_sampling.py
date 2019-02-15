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

import random

"""
Brief:
From a given query, randomly or pseudo-randomly take N of the answers. If there are less than N results, then sample 
with replacement. 
-Ideally don't require asking for all answers to save time.
-We want the results to be streamed

"""

random.seed(1)


def random_sample(population, sample_size):
    """
    Samples items from an iterable population retrieving `sample_size` samples randomly without replacement. If
    `sample_size` samples aren't available only yield the first `sample_size` items
    :param population: An iterator of items to sample
    :param sample_size: The number of items to randomly retrieve from the population
    :return: A list of randomly selected items, possibly containing duplicates
    """
    results = []
    empty = True

    for i, item in enumerate(population):
        empty = False
        r = random.randint(0, i)
        if r < sample_size:
            if i < sample_size:
                results.insert(r, item)  # add first n items in random order
            else:
                results[r] = item  # at a decreasing rate, replace random items

    if empty:
        raise ValueError('Population is empty')

    return results
