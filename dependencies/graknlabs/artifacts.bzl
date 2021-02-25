#
# Copyright (C) 2020 Grakn Labs
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

load("@graknlabs_dependencies//distribution/artifact:rules.bzl", "native_artifact_files")
load("@graknlabs_dependencies//distribution:deployment.bzl", "deployment")

def graknlabs_grakn_core_artifacts():
    native_artifact_files(
        name = "graknlabs_grakn_core_artifact",
        group_name = "graknlabs_grakn_core",
        artifact_name = "grakn-core-server-{platform}-{version}.{ext}",
        tag_source = deployment["artifact.release"],
        commit_source = deployment["artifact.snapshot"],
        commit = "118eee244a4949c629de0277804155ebd4b316be",
    )