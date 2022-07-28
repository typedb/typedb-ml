#
#  Copyright (C) 2022 Vaticle
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
from typedb.api.connection.transaction import TransactionType


def get_thing_types(session):
    """
    Get all schema types, excluding those for implicit attribute relations and base types
    Args:
        session: TypeDB session

    Returns:
        TypeDB types
    """
    with session.transaction(TransactionType.READ) as tx:
        schema_concepts = tx.query().match("match $x sub thing;")
        thing_types = [schema_concept.get('x').get_label().name() for schema_concept in schema_concepts]
    [thing_types.remove(el) for el in ['thing', 'relation', 'entity', 'attribute']]
    return thing_types


def get_role_triplets(tx):
    """
    Get triples of all schema roles and the relation and roleplayer they connect
    Args:
        tx: TypeDB transaction

    Returns:
        TypeDB role triples
    """
    role_triples = []
    schema_concepts = tx.query().match("match $rel sub relation; $rel relates $r; $rp plays $r;")
    for answer in schema_concepts:
        relation = answer.get('rel').get_label().name()
        roles = [r.get_label().name() for r in answer.get('r').as_remote(tx).get_supertypes()]
        roles.remove('role')
        player = answer.get("rp").get_label().name()
        for role in roles:
            role_triples.append((relation, role, player))
    return role_triples


def get_has_triplets(tx):
    """
    Get triples of all ownerships: the owner type and owned attribute
    Args:
        tx: TypeDB transaction

    Returns:
        TypeDB ownership triples
    """
    has_triples = []
    schema_concepts = tx.query().match("match $owner sub thing, owns $owned;")
    for answer in schema_concepts:
        owner = answer.get('owner').get_label().name()
        owned = answer.get('owned').get_label().name()
        has_triples.append((owner, "has", owned))
    return has_triples


def get_edge_type_triplets(session):
    with session.transaction(TransactionType.READ) as tx:
        edge_types = get_role_triplets(tx) + get_has_triplets(tx)
    return edge_types


def reverse_edge_type_triplets(edge_types):
    reversed_edge_triples = []
    for edge_from, edge, edge_to in edge_types:
        reversed_edge_triples.append((edge_to, f"rev_{edge}", edge_from))
    return reversed_edge_triples
