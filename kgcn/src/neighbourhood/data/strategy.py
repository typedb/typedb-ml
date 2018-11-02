
TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target


class DataTraversalStrategy:

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

    def __init__(self, roles_played_query=ROLES_PLAYED_QUERY, roleplayers_query=ROLEPLAYERS_QUERY):
        """
        Strategy to determine how the knowledge graph is traversed. Used to store parameters
        """
        self.roles_played_query = roles_played_query
        self.roleplayers_query = roleplayers_query


