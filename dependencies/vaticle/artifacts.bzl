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

load("@vaticle_dependencies//distribution/artifact:rules.bzl", "native_artifact_files")
load("@vaticle_dependencies//distribution:deployment.bzl", "deployment")

def vaticle_typedb_artifacts():
    native_artifact_files(
        name = "vaticle_typedb_artifact",
        group_name = "vaticle_typedb",
        artifact_name = "typedb-server-{platform}-{version}.{ext}",
        tag_source = deployment["artifact.release"],
        commit_source = deployment["artifact.snapshot"],
        commit = "447615e8a612c2a071dee1f2f9061776598892f2",
    )