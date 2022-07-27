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

load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_typedb_ml_pip//:requirements.bzl", vaticle_typedb_ml_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "diagnosis",
    srcs = [
        'diagnosis.py'
    ],
    deps = [
        "//examples/diagnosis/dataset",
        "//typedb_ml/pytorch_geometric",
        "//typedb_ml/networkx",
        "//typedb_ml/typedb",
        "@vaticle_typedb_client_python//:client_python",
        vaticle_typedb_ml_requirement("networkx"),
        vaticle_typedb_ml_requirement("torch"),
        vaticle_typedb_ml_requirement("torch_geometric"),
        vaticle_typedb_ml_requirement("torch_sparse"),
        vaticle_typedb_ml_requirement("torch_scatter"),
        vaticle_typedb_ml_requirement("tensorboard"),
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    exclude = glob([
        "*.md"
    ]),
    license_type = "apache-header",
)
