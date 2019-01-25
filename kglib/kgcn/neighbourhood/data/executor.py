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

import kglib.kgcn.neighbourhood.data.utils as utils


TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target

ROLES_PLAYED = 0
ROLEPLAYERS = 1

DATA_TYPE_NAMES = ('long', 'double', 'boolean', 'date', 'string')


class TraversalExecutor:

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
        'query': "match $thing id {}; $relationship id {}; $relationship($role: $thing); get $role;",
        'variable': 'role'
    }

    ROLES_PLAYED_QUERY = {
        'query': "match $thing id {}; $relationship($thing); get $relationship, $thing;",
        'role_direction': TARGET_PLAYS,
        'target_variable': 'thing',
        'neighbour_variable': 'relationship'}

    ROLEPLAYERS_QUERY = {
        'query': "match $relationship id {}; $relationship($thing); get $thing, $relationship;",
        'role_direction': NEIGHBOUR_PLAYS,
        'target_variable': 'relationship',
        'neighbour_variable': 'thing'}

    def __init__(self, find_neighbours_from_attributes=True, attributes_via_implicit_relationships=False,
                 roles_played_query=ROLES_PLAYED_QUERY, roleplayers_query=ROLEPLAYERS_QUERY):
        self._find_neighbours_from_attributes = find_neighbours_from_attributes
        self._attributes_via_implicit_relationships = attributes_via_implicit_relationships
        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query

    def _query(self, query, tx):
        print(query)
        return tx.query(query)

    def __call__(self, query_direction, concept_id, tx):
        """
        Takes a query to execute and the variables to return
        :param query_direction: whether we want to retrieve roles played or role players
        :param concept_id: id for the concept to find connections for
        :return:
        """

        if query_direction == ROLES_PLAYED:
            base_query = self.ROLES_PLAYED_QUERY
            thing_variable = base_query['target_variable']
            relationship_variable = base_query['neighbour_variable']

        elif query_direction == ROLEPLAYERS:
            base_query = self.ROLEPLAYERS_QUERY
            thing_variable = base_query['neighbour_variable']
            relationship_variable = base_query['target_variable']
        else:
            raise ValueError('query_direction isn\'t properly defined')

        query = base_query['query'].format(concept_id)
        print(query)
        connection_iterator = self._query(query, tx)

        def _roles_iterator():

            # Direct connections to attributes
            if not self._attributes_via_implicit_relationships:

                target_query = self.TARGET_QUERY['query'].format(concept_id)
                target_concept = next(self._query(target_query, tx)).get(self.TARGET_QUERY['variable'])

                if target_concept.type().is_implicit():
                    raise ValueError(
                        "A target concept has been found to be implicit, but using implicit relationships has "
                        "been optionally disabled")
                attribute_query = self.ATTRIBUTE_QUERY['query'].format(concept_id)
                attributes = map(lambda x: x.get(self.ATTRIBUTE_QUERY['variable']), self._query(attribute_query, tx))

                for attribute in attributes:
                    neighbour_info = build_concept_info(attribute)
                    yield {'role_label': self.ATTRIBUTE_ROLE_LABEL, 'role_direction': NEIGHBOUR_PLAYS,
                           'neighbour_info': neighbour_info}

                if target_concept.is_attribute() and self._find_neighbours_from_attributes:
                    attribute_owners_query = self.ATTRIBUTE_OWNER_QUERY['query'].format(target_concept.id)
                    neighbours = map(lambda x: x.get(self.ATTRIBUTE_OWNER_QUERY['variable']),
                                     self._query(attribute_owners_query, tx))

                    for neighbour in neighbours:
                        neighbour_info = build_concept_info(neighbour)
                        yield {'role_label': self.ATTRIBUTE_ROLE_LABEL, 'role_direction': TARGET_PLAYS,
                               'neighbour_info': neighbour_info}

            # Connections to entities, relationships and optionally implicit relationships
            for answer in connection_iterator:
                relationship = answer.get(relationship_variable)
                thing = answer.get(thing_variable)
                if (relationship.type().is_implicit() or thing.type().is_implicit()) and not self._attributes_via_implicit_relationships:
                    pass
                else:

                    role_sups = self._find_roles(thing, relationship, tx)
                    role = find_lowest_role_from_rols_sups(role_sups)

                    role_label = role.label()

                    neighbour_concept = answer.get(base_query['neighbour_variable'])
                    neighbour_info = build_concept_info(neighbour_concept)

                    yield {'role_label': role_label, 'role_direction': base_query['role_direction'],
                           'neighbour_info': neighbour_info}

        return _roles_iterator()

    def _find_roles(self, thing, relationship, tx):
        query_str = self.ROLE_QUERY['query'].format(thing.id, relationship.id)
        answers = self._query(query_str, tx)
        role_sups = [answer.get(self.ROLE_QUERY['variable']) for answer in answers]
        return role_sups


def find_lowest_role_from_rols_sups(role_sups):
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


class ConceptInfo(utils.PropertyComparable):
    def __init__(self, id, type_label, base_type_label, data_type=None, value=None):
        self.id = id
        self.type_label = type_label
        self.base_type_label = base_type_label  # TODO rename to base_type in line with Client Python

        # If the concept is an attribute
        self.data_type = data_type
        self.value = value


def build_concept_info(concept):

    id = concept.id
    type_label = concept.type().label()
    base_type_label = concept.base_type.lower()

    assert(base_type_label in ['entity', 'relationship', 'attribute'])

    if base_type_label == 'attribute':
        data_type = concept.type().data_type().name.lower()
        assert data_type in DATA_TYPE_NAMES
        value = concept.value()

        return ConceptInfo(id, type_label, base_type_label, data_type, value)

    return ConceptInfo(id, type_label, base_type_label)
