#
# Copyright (C) 2022 Vaticle
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_typedb_ml_pip//:requirements.bzl", vaticle_typedb_ml_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "networkx",
    srcs = [
        'query_graph.py',
        'concept_dict_to_networkx.py',
        'queries_to_networkx.py',
        'iterate.py',
    ],
    deps = [
        "//typedb_ml/typedb",
        vaticle_typedb_ml_requirement('networkx'),
        vaticle_typedb_ml_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)

py_test(
    name = "query_graph_test",
    srcs = [
        "query_graph_test.py",
        "query_graph.py"
    ],
    deps = [
        vaticle_typedb_ml_requirement('networkx'),
    ]
)

py_test(
    name = "queries_to_networkx_test",
    srcs = [
        "queries_to_networkx_test.py",
        "queries_to_networkx.py",
        "concept_dict_to_networkx.py",
        "query_graph.py",
        "graph_test_case.py"
    ],
    deps = [
        "//typedb_ml/typedb",
        vaticle_typedb_ml_requirement('networkx'),
    ]
)

py_test(
    name = "queries_to_networkx_it",
    srcs = [
        "queries_to_networkx_it.py",
        "queries_to_networkx.py",
        "concept_dict_to_networkx.py",
        "query_graph.py",
        "graph_test_case.py"
    ],
    deps = [
        "//typedb_ml/typedb",
        "//typedb_ml/typedb/test",
        "@vaticle_typedb_client_python//:client_python",
        vaticle_typedb_ml_requirement('networkx'),
    ],
    data = ["@vaticle_typedb_artifact_linux//file"],
    args = ["$(location @vaticle_typedb_artifact_linux//file)"],
)


py_test(
    name = "concept_dict_to_networkx_test",
    srcs = [
        "concept_dict_to_networkx_test.py",
        "concept_dict_to_networkx.py",
        "graph_test_case.py"
    ],
    deps = [
        "//typedb_ml/typedb",
        vaticle_typedb_ml_requirement('networkx'),
    ],
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache-header",
)
