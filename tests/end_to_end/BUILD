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

load("@rules_python//python:defs.bzl", "py_test")
load("@vaticle_bazel_distribution//artifact:rules.bzl", "artifact_extractor")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_test(
    name = "diagnosis",
    size = "enormous",
    srcs = [
        "diagnosis.py"
    ],
    deps = [
        "//examples/diagnosis",
        "//typedb_ml/typedb/test",
    ],
    data = [
        "//examples/diagnosis/dataset:diagnosis-example-typeql-files",
        "@vaticle_typedb_artifact_linux//file"
    ],
    args = [
        "$(locations //examples/diagnosis/dataset:diagnosis-example-typeql-files)",
        "$(location @vaticle_typedb_artifact_linux//file)",
    ],
)

py_test(
    name = "diagnosis-debug",
    size = "enormous",
    srcs = [
        "diagnosis_debug.py"
    ],
    main = "diagnosis_debug.py",
    deps = [
        "//examples/diagnosis",
    ],
    data = [
        "//examples/diagnosis/dataset:diagnosis-example-typeql-files",
    ],
    args = [
        "$(locations //examples/diagnosis/dataset:diagnosis-example-typeql-files)",
    ],
)

artifact_extractor(
    name = "typedb-extractor-linux",
    artifact = "@vaticle_typedb_artifact_linux//file",
)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
    ]),
    license_type = "apache-header",
)
