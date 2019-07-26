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


class ExampleGenerator:
    def __init__(self, query_feature_fns, pmf):
        self._pmf = pmf
        self._query_feature_fns = query_feature_fns
        self._example_id = 0

    def generate_example(self):
        variable_values = self._pmf.select()

        feature_queries = []

        for feature_fn in self._query_feature_fns:
            queries = feature_fn(variable_values, self._example_id)
            if queries:
                feature_queries.extend(queries)

        self._example_id += 1
        return feature_queries
