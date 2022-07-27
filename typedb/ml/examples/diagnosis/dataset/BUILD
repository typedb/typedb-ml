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
    name = "dataset",
    srcs = [
        'generate.py',
        'pmf.py',
    ],
    deps = [
        "@vaticle_typedb_client_python//:client_python",
        vaticle_typedb_ml_requirement('numpy'),
        vaticle_typedb_ml_requirement('pandas'),
        vaticle_typedb_ml_requirement('pytz'),
        vaticle_typedb_ml_requirement('python-dateutil'),
        vaticle_typedb_ml_requirement('six')
    ],
    visibility=['//visibility:public']
)

filegroup(
    name = "diagnosis-example-typeql-files",
    srcs = [
        "schema.tql",
        "seed_data.tql",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "pmf_test",
    srcs = [
        "pmf_test.py"
    ],
    deps = [
        "dataset"
    ]
)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
    ]),
    license_type = "apache-header",
)
