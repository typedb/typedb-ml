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

import kglib.kgcn.core.ingest.traverse.schema.traversal as trav


class TraversalExecutor:
    def __init__(self, grakn_tx):
        self._grakn_tx = grakn_tx

    def _query(self, query):
        print(query)
        return self._grakn_tx.query(query)

    def get_schema_concept_types(self, get_types_query, include_implicit=False, include_metatypes=False):

        for answer in self._query(get_types_query):
            t = answer.get('x')

            if not (((not include_implicit) and t.is_implicit()) or (
                    (not include_metatypes) and t.label() in trav.METATYPE_LABELS)):
                yield t
