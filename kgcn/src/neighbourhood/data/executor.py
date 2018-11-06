import kgcn.src.neighbourhood.data.utils as utils


TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target

ROLES_PLAYED = 0
ROLEPLAYERS = 1


class TraversalExecutor:

    # RELATIONSHIP_VARIABLE = 'relationship'
    # THING_VARIABLE = 'thing'

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

    def __init__(self, grakn_tx, roles_played_query=ROLES_PLAYED_QUERY, roleplayers_query=ROLEPLAYERS_QUERY):
        self._grakn_tx = grakn_tx
        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query

    def __call__(self, query_direction, concept_id):
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
        connection_iterator = self._grakn_tx.query(query)

        def _roles_iterator():
            for answer in connection_iterator:
                relationship = answer.get(relationship_variable)
                thing = answer.get(thing_variable)

                role_label = find_shared_role_label(thing, relationship)
                neighbour_concept = answer.get(base_query['neighbour_variable'])
                neighbour_info = build_concept_info(neighbour_concept)

                yield {'role_label': role_label, 'role_direction': base_query['role_direction'],
                       'neighbour_info': neighbour_info}

        return _roles_iterator()


def find_shared_role_label(thing_instance, relationship_instance):
    """
    Make use of the schema to see if there is a single role relates a thing to a relationship
    :param thing_instance: instance of a thing that we believe plays a role in a relationship
    :param relationship_instance: instance of a relationship we believe thing plays a role in
    :return:
    """
    roles_thing_plays = set(map(lambda x: x.label(), thing_instance.type().playing()))
    roles_relationship_relates = set(map(lambda x: x.label(), relationship_instance.type().roles()))
    intersection = roles_thing_plays.intersection(roles_relationship_relates)
    assert(len(intersection) == 1)
    return intersection.pop()


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
        assert data_type in ('long', 'double', 'boolean', 'date', 'string')
        value = concept.value()

        return ConceptInfo(id, type_label, base_type_label, data_type, value)

    return ConceptInfo(id, type_label, base_type_label)
