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

import collections

METATYPE_LABELS = ['thing', 'entity', 'relation', 'attribute', 'role', '@has-attribute-value',
                   '@has-attribute-owner', '@has-attribute']


def labels_from_types(schema_concept_types):
    for type in schema_concept_types:
        yield type.label()


def get_sups_labels_per_type(schema_concept_types, include_metatypes=False, include_self=False):

    schema_concept_super_types = collections.OrderedDict()

    for schema_concept_type in schema_concept_types:
        super_types = schema_concept_type.sups()

        super_type_labels = []
        for super_type in super_types:
            super_type_label = super_type.label()

            if not (((not include_self) and super_type_label == schema_concept_type.label()) or (
                (not include_metatypes) and super_type.label() in METATYPE_LABELS)):
                super_type_labels.append(super_type.label())
        schema_concept_super_types[schema_concept_type.label()] = super_type_labels
    return schema_concept_super_types


def traverse_schema(traversal_executor, query):
    schema_concept_types = list(traversal_executor.get_schema_concept_types(query))

    schema_concept_super_type_labels = get_sups_labels_per_type(schema_concept_types, include_self=True)
    return schema_concept_super_type_labels
