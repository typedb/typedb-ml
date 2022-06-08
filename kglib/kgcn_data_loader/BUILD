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

load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

py_library(
    name = "kgcn_data_loader",
    srcs = glob([
        "*.py",
        "**/*.py"
    ]),
    visibility=['//visibility:public']
)

py_test(
    name = "standard_encode_test",
    srcs = [
        "encoding/standard_encode_test.py"
    ],
    deps = [
        "kgcn_data_loader",
        "//kglib/utils"
    ]
)

py_test(
    name = "utils_test",
    srcs = [
        "utils_test.py"
    ],
    deps = [
        "kgcn_data_loader",
        "//kglib/utils/graph/test",
        "//kglib/utils/typedb",
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('numpy'),
    ]
)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
        "**/*"
    ]),
    license_type = "apache",
)
