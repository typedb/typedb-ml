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

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def vaticle_dependencies():
    git_repository(
        name = "vaticle_dependencies",
        remote = "https://github.com/vaticle/dependencies",
        commit = "6904ecb83c744097ef993f29f9941d55538ab331",  # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_dependencies
    )

def vaticle_typedb_client_python():
    git_repository(
        name = "vaticle_typedb_client_python",
        remote = "https://github.com/vaticle/typedb-client-python",
        commit = "38a359e5607a66c3490eb9772dd5ef485d06a5bd" # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_typedb_client_python
    )

def vaticle_common():
    git_repository(
        name = "vaticle_common",
        remote = "https://github.com/vaticle/typedb-common",
        commit = "77d07410401435e8274e82c07b73437469e290fe" # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_common
    )
