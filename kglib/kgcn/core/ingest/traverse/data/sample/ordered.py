
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

def ordered_sample(population, sample_size):
    """
    Samples the population, taking the first `n` items (a.k.a `sample_size') encountered. If more samples are
    requested than are available then only yield the first `sample_size` items
    :param population: An iterator of items to sample
    :param sample_size: The number of items to retrieve from the population
    :return: A list of the first `sample_size` items from the population
    """

    empty = True
    results = []
    for i, item in enumerate(population):
        empty = False
        if i >= sample_size:
            break
        results.append(item)

    if empty:
        raise ValueError('Population is empty')

    return results
