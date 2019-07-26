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


def generate_example_queries(feature_value_selector, query_feature_fns, example_id):
    """
    Generates Graql queries that can be used to create an example, where an example is a subgraph intended to be
    taken as input training/test data to ML algorithms.

    Args:
        feature_value_selector: An object used to select the values of all of the features
        query_feature_fns: A list of functions that all accept `variable_values` and `example_id`, and based on these
            return a list of queries
        example_id: The unique id to use for this example, likely used by the `query_feature_fns` to uniquely identify
            concepts belonging to a particular example

    Returns: A list of Graql queries which when executed sequentially will create an example
    """

    variable_values = feature_value_selector.select()

    feature_queries = []

    for feature_fn in query_feature_fns:
        queries = feature_fn(variable_values, example_id)
        if queries:
            feature_queries.extend(queries)

    return feature_queries
