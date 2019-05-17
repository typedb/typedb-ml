#
#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
#

import kglib.kgcn.core.ingest.traverse.data.context.neighbour as neighbour


def gen(elements):
    for el in elements:
        yield el


def _build_data(role_label, role_direction, neighbour_id, neighbour_type, neighbour_metatype, data_type=None,
                value=None):
    return neighbour.Connection(role_label, role_direction,
                                neighbour.Thing(neighbour_id, neighbour_type, neighbour_metatype, data_type=data_type,
                                                value=value))


class MockNeighbourFinder:

    def find(self, thing_id, tx):

        if thing_id == "0":  # person

            yield from gen([
                _build_data("has", neighbour.NEIGHBOUR_PLAYS, "1", "name", "attribute", data_type='string', value="Sundar Pichai"),
                _build_data("employee", neighbour.TARGET_PLAYS, "2", "employment", "relation"),
            ])

        elif thing_id == "1":  # person's name

            yield from gen([_build_data("has", neighbour.TARGET_PLAYS, "0", "person", "entity")])

        elif thing_id == "2":  # employment relation

            yield from gen([_build_data("employer", neighbour.NEIGHBOUR_PLAYS, "3", "company", "entity"),
                            _build_data("employee", neighbour.NEIGHBOUR_PLAYS, "0", "person", "entity")])

        elif thing_id == "3":  # company name

            yield from gen([_build_data("has", neighbour.NEIGHBOUR_PLAYS, "4", "name", "attribute",
                                        data_type='string', value="Google")])

        else:
            raise ValueError("This concept id hasn't been mocked")
