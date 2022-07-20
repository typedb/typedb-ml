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
        commit = "a8b3b714a5e0562d41f4c05ca8f266d48b7d0fd3",  # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_dependencies
    )

def vaticle_typedb_client_python():
    git_repository(
        name = "vaticle_typedb_client_python",
        remote = "https://github.com/vaticle/typedb-client-python",
        commit = "d08a6c36ac4ef20f2df26017cf42e17fba230387" # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_typedb_client_python
    )

def vaticle_typedb_common():
    git_repository(
        name = "vaticle_typedb_common",
        remote = "https://github.com/vaticle/typedb-common",
        commit = "f0dd708adaea9fe1fdc3699180797a12166d33e8" # sync-marker: do not remove this comment, this is used for sync-dependencies by @vaticle_common
    )
