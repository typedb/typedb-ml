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

import kglib.kgcn.core.ingest.traverse.data.context.utils as utils


TARGET_PLAYS = 0  # In this case, the neighbour is a relation in which this thing plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target

DATA_TYPE_NAMES = ('long', 'double', 'boolean', 'date', 'string')


class NeighbourFinder:

    TARGET_QUERY = {
        'query': 'match $target id {}; get;',
        'variable': 'target'
    }

    ATTRIBUTE_QUERY = {
        'query': 'match $thing id {}, has attribute $attribute; get $attribute;',
        'variable': 'attribute'
    }

    ATTRIBUTE_OWNER_QUERY = {
        'query': 'match $attribute-owner has attribute $a; $a id {}; get;',
        'variable': 'attribute-owner'
    }

    ATTRIBUTE_ROLE_LABEL = 'has'

    ROLE_QUERY = {
        'query': "match $thing id {}; $relation id {}; $relation($role: $thing); get $role;",
        'variable': 'role'
    }

    ROLES_PLAYED_QUERY = {
        'query': "match $thing id {}; $relation($thing); get $relation, $thing;",
        'role_direction': TARGET_PLAYS,
        'target_variable': 'thing',
        'neighbour_variable': 'relation'}

    ROLEPLAYERS_QUERY = {
        'query': "match $relation id {}; $relation($thing); get $thing, $relation;",
        'role_direction': NEIGHBOUR_PLAYS,
        'target_variable': 'relation',
        'neighbour_variable': 'thing'}

    def __init__(self, find_neighbours_from_attributes=True, roles_played_query=ROLES_PLAYED_QUERY,
                 roleplayers_query=ROLEPLAYERS_QUERY):
        self._find_neighbours_from_attributes = find_neighbours_from_attributes
        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query

    def _query(self, query, tx):
        print(query)
        return tx.query(query)

    def find(self, thing_id, tx):
        """
        Takes a query to execute and the variables to return
        :param thing_id: id for the thing to find connections for
        :return:
        """

        def _link_iterator():
            target_query = self.TARGET_QUERY['query'].format(thing_id)
            target_thing = next(self._query(target_query, tx)).get(self.TARGET_QUERY['variable'])

            # Direct connections to attributes
            yield from self._find_direct_attribute_neighbours(tx, target_thing, thing_id)

            if target_thing.is_attribute() and self._find_neighbours_from_attributes:
                yield from self._find_neighbours_from_attribute(tx, target_thing)

            # # Connections to entities, relations and optionally implicit relations
            # yield from self._find_entity_and_relation_neighbours(query_direction, thing_id, tx)

            yield from self._find_neighbour_relations_where_thing_plays_role(tx, thing_id)

            if target_thing.is_relation():
                yield from self._find_neighbour_roleplayers(tx, thing_id)

        return _link_iterator()

    def _find_neighbour_relations_where_thing_plays_role(self, tx, thing_id):
        base_query = self.ROLES_PLAYED_QUERY
        thing_variable = base_query['target_variable']
        relation_variable = base_query['neighbour_variable']
        yield from self._get_role_link(tx, base_query, relation_variable, thing_variable, thing_id)

    def _find_neighbour_roleplayers(self, tx, thing_id):
        base_query = self.ROLEPLAYERS_QUERY
        thing_variable = base_query['neighbour_variable']
        relation_variable = base_query['target_variable']
        yield from self._get_role_link(tx, base_query, relation_variable, thing_variable, thing_id)

    def _get_role_link(self, tx, base_query, relation_variable, thing_variable, thing_id):
        query = base_query['query'].format(thing_id)
        link_iterator = self._query(query, tx)

        for answer in link_iterator:
            relation = answer.get(relation_variable)
            thing = answer.get(thing_variable)
            if not(relation.type().is_implicit() or thing.type().is_implicit()):
                role_sups = self._find_roles(thing, relation, tx)
                role = find_lowest_role_from_role_sups(role_sups)

                role_label = role.label()

                neighbour_grakn_thing = answer.get(base_query['neighbour_variable'])
                neighbour_thing = build_thing(neighbour_grakn_thing)

                yield {'role_label': role_label, 'role_direction': base_query['role_direction'],
                       'neighbour_thing': neighbour_thing}

    def _find_roles(self, thing, relation, tx):
        query_str = self.ROLE_QUERY['query'].format(thing.id, relation.id)
        answers = self._query(query_str, tx)
        role_sups = [answer.get(self.ROLE_QUERY['variable']) for answer in answers]
        return role_sups

    def _find_direct_attribute_neighbours(self, tx, target_thing, thing_id):

        if target_thing.type().is_implicit():
            raise ValueError(
                "A target thing has been found to be implicit, but using implicit relations has "
                "been optionally disabled")

        attribute_query = self.ATTRIBUTE_QUERY['query'].format(thing_id)
        attributes = map(lambda x: x.get(self.ATTRIBUTE_QUERY['variable']), self._query(attribute_query, tx))

        for attribute in attributes:
            neighbour_thing = build_thing(attribute)
            yield {'role_label': self.ATTRIBUTE_ROLE_LABEL, 'role_direction': NEIGHBOUR_PLAYS,
                   'neighbour_thing': neighbour_thing}

    def _find_neighbours_from_attribute(self, tx, target_thing):

        attribute_owners_query = self.ATTRIBUTE_OWNER_QUERY['query'].format(target_thing.id)
        neighbours = map(lambda x: x.get(self.ATTRIBUTE_OWNER_QUERY['variable']),
                         self._query(attribute_owners_query, tx))

        for neighbour in neighbours:
            neighbour_thing = build_thing(neighbour)
            yield {'role_label': self.ATTRIBUTE_ROLE_LABEL, 'role_direction': TARGET_PLAYS,
                   'neighbour_thing': neighbour_thing}


def find_lowest_role_from_role_sups(role_sups):
    """
    Take a list containing a hierarchy of role concepts (order not necessarily known), and find the 'lowest' role.
    That is, the role without any of it's subtypes in the list.
    :param role_sups: list of role supertypes
    :return: role without subtypes present (lowest role)
    """

    role_sups_labels = [role.label() for role in role_sups]

    while len(role_sups_labels) > 0:
        label = role_sups_labels.pop(0)
        role = role_sups.pop(0)
        if len({s.label() for s in role.subs()}.intersection(set(role_sups_labels))) == 0:
            break
    else:
        raise ValueError
    return role


class Thing(utils.PropertyComparable):
    def __init__(self, id, type_label, base_type_label, data_type=None, value=None):
        self.id = id
        self.type_label = type_label
        self.base_type_label = base_type_label  # TODO rename to base_type in line with Client Python

        # If the thing is an attribute
        self.data_type = data_type
        self.value = value


def build_thing(grakn_thing):

    id = grakn_thing.id
    type_label = grakn_thing.type().label()
    base_type_label = grakn_thing.base_type.lower()

    assert(base_type_label in ['entity', 'relation', 'attribute'])

    if base_type_label == 'attribute':
        data_type = grakn_thing.type().data_type().name.lower()
        assert data_type in DATA_TYPE_NAMES
        value = grakn_thing.value()

        return Thing(id, type_label, base_type_label, data_type, value)

    return Thing(id, type_label, base_type_label)
