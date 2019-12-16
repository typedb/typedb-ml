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
        tx: Grakn transaction

    Returns:
        Grakn types
    """
    schema_concepts = tx.query(
        "match $x sub thing; "
        "not {$x sub @has-attribute;}; "
        "not {$x sub @key-attribute;}; "
        "get;")
    thing_types = [schema_concept.get('x').label() for schema_concept in schema_concepts]
    [thing_types.remove(el) for el in ['thing', 'relation', 'entity', 'attribute']]
    return thing_types


def get_role_types(tx):
    """
    Get all schema roles, excluding those for implicit attribute relations, the base role type
    Args:
        tx: Grakn transaction

    Returns:
        Grakn roles
    """
    schema_concepts = tx.query(
        "match $x sub role; "
        "not{$x sub @key-attribute-value;}; "
        "not{$x sub @key-attribute-owner;}; "
        "not{$x sub @has-attribute-value;}; "
        "not{$x sub @has-attribute-owner;};"
        "get;")
    role_types = ['has'] + [role.get('x').label() for role in schema_concepts]
    role_types.remove('role')
    return role_types
