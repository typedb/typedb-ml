#
#  Copyright (C) 2021 Vaticle
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

def get_thing_types(tx):
    """
    Get all schema types, excluding those for implicit attribute relations and base types
    Args:
        tx: TypeDB transaction

    Returns:
        TypeDB types
    """
    schema_concepts = tx.query().match("match $x sub thing;")
    thing_types = [schema_concept.get('x').get_label().name() for schema_concept in schema_concepts]
    [thing_types.remove(el) for el in ['thing', 'relation', 'entity', 'attribute']]
    return thing_types


def get_role_types(tx):
    """
    Get all schema roles, excluding those for implicit attribute relations, the base role type
    Args:
        tx: TypeDB transaction

    Returns:
        TypeDB roles
    """
    schema_concepts = tx.query().match("match $rel sub relation, relates $r;")
    role_types = ['has'] + [role.get('r').get_label().name() for role in schema_concepts]
    role_types.remove('role')
    return role_types
