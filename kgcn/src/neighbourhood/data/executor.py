import kgcn.src.neighbourhood.data.concept as concept

UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL = "UNKNOWN_ROLE_NEIGHBOUR_PLAYS"
UNKNOWN_ROLE_TARGET_PLAYS_LABEL = "UNKNOWN_ROLE_TARGET_PLAYS"

TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target

ROLES_PLAYED = 0
ROLEPLAYERS = 1


class TraversalExecutor:

    # TODO Changing queries due to bug
    # query = "match $x id {}; $relationship($role: $x); get $relationship, $role;")
    ROLES_PLAYED_QUERY = {
        'query': "match $x id {}; $relationship($x); get $relationship;",
        'role_variable': 'role',
        'role_direction': TARGET_PLAYS,
        'neighbour_variable': 'relationship'}

    # query = "match $relationship id {}; $relationship($role: $x) isa {}; get $x, $role;"
    ROLEPLAYERS_QUERY = {
        'query': "match $relationship id {}; $relationship($x) isa {}; get $x;",
        'role_variable': 'role',
        'role_direction': NEIGHBOUR_PLAYS,
        'neighbour_variable': 'x'}

    def __init__(self, grakn_tx, roles_played_query=ROLES_PLAYED_QUERY, roleplayers_query=ROLEPLAYERS_QUERY):
        self._grakn_tx = grakn_tx
        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query

    def __call__(self, query_direction, *args):
            """
            Takes a query to execute and the variables to return
            :param query_direction: whether we want to retrieve roles played or role players
            :param concept_id: id for the concept to find connections for
            :return:
            """

            if query_direction == ROLES_PLAYED:
                base_query = self.ROLES_PLAYED_QUERY
            elif query_direction == ROLEPLAYERS:
                base_query = self.ROLEPLAYERS_QUERY
            else:
                raise ValueError('query_direction isn\'t properly defined')

            query = base_query['query'].format(*args)
            print(query)
            roles_iterator = self._grakn_tx.query(query)

            def _roles_iterator():
                for answer in roles_iterator:
                    # TODO See above, omitting due to bug
                    # role_label = answer.get(base_query['role_variable']).label()
                    role_label = UNKNOWN_ROLE_TARGET_PLAYS_LABEL
                    relationship_concept = answer.get(base_query['neighbour_variable'])
                    relationship_info = concept.build_concept_info(relationship_concept)

                    yield {'role_label': role_label, 'role_direction': base_query['role_direction'],
                           'neighbour_info': relationship_info}

            return _roles_iterator()
