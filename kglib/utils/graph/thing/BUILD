#
# Copyright (C) 2021 Vaticle
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

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")


py_test(
    name = "queries_to_networkx_graph_test",
    srcs = [
        "queries_to_networkx_graph_test.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/typedb/test",
        "//kglib/utils/graph/test",
    ]
)

py_test(
    name = "queries_to_networkx_graph_it",
    srcs = [
        "queries_to_networkx_graph_it.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/typedb/test",
        "//kglib/utils/graph/test",
        "@vaticle_typedb_client_python//:client_python",
    ],
    data = ["@vaticle_typedb_artifact_linux//file"],
    args = ["$(location @vaticle_typedb_artifact_linux//file)"],
)


py_test(
    name = "concept_dict_to_graph_test",
    srcs = [
        "concept_dict_to_graph_test.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/graph/test",
    ],
)

py_library(
    name = "thing",
    srcs = [
        'concept_dict_to_networkx_graph.py',
        'queries_to_networkx_graph.py'
    ],
    deps = [
        "//kglib/utils/typedb/object",
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
