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

exports_files(["requirements.txt", "RELEASE_TEMPLATE.md"])

load("@rules_python//python:defs.bzl", "py_library", "py_test")

load("@vaticle_typedb_ml_pip//:requirements.bzl", vaticle_typedb_ml_requirement = "requirement")

load("@vaticle_bazel_distribution//github:rules.bzl", "deploy_github")
load("@vaticle_bazel_distribution//pip:rules.bzl", "assemble_pip", "deploy_pip")
load("@vaticle_typedb_ml_pip//:requirements.bzl", vaticle_typedb_ml_requirement = "requirement")

load("@vaticle_dependencies//distribution:deployment.bzl", "deployment")
load("//:deployment.bzl", github_deployment = "deployment")
load("@vaticle_dependencies//tool/release/deps:rules.bzl", "release_validate_python_deps")

load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

assemble_pip(
    name = "assemble-pip",
    target = "//typedb_ml:typedb-ml",
    package_name = "typedb-ml",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    url = "https://github.com/vaticle/typedb-ml",
    author = "Vaticle",
    author_email = "community@vaticle.com",
    license = "Apache-2.0",
    requirements_file = "//:install_requires.txt",
    keywords = ["machine learning", "logical reasoning", "knowledege graph", "typedb", "database", "graph",
                "knowledgebase", "knowledge-engineering"],

    description = "A Machine Learning Library for TypeDB.",
    long_description_file = "//:README.md",
)

deploy_pip(
    name = "deploy-pip",
    target = ":assemble-pip",
    snapshot = deployment["pypi.snapshot"],
    release = deployment["pypi.release"],
)

deploy_github(
    name = "deploy-github",
    release_description = "//:RELEASE_TEMPLATE.md",
    title = "TypeDB-ML",
    title_append_version = True,
    organisation = github_deployment["github.organisation"],
    repository = github_deployment["github.repository"],
    draft = False
)

release_validate_python_deps(
    name = "release-validate-python-deps",
    requirements = "//:requirements.txt",
    tagged_deps = [
        "typedb-client",
    ],
)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
        ".factory/*",
    ]),
    exclude = glob([
        "*.md"
    ]) + [
        ".bazelversion",
        "LICENSE",
        "VERSION",
    ],
    license_type = "apache-header",
)

checkstyle_test(
    name = "checkstyle-license",
    include = ["LICENSE"],
    license_type = "apache-fulltext",
)
