import kgcn.src.neighbourhood.data.concept as concept

UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL = "UNKNOWN_ROLE_NEIGHBOUR_PLAYS"
UNKNOWN_ROLE_TARGET_PLAYS_LABEL = "UNKNOWN_ROLE_TARGET_PLAYS"


class TraversalExecutor:
    def __init__(self, grakn_tx):
        self._grakn_tx = grakn_tx

    def get_neighbour_connections(self, query, role_variable, role_direction, neighbour_variable):
        """
        Takes a query to execute and the variables to return
        :param query: query to execute
        :param role_variable: variable representing the role connecting two concepts
        :param role_direction: which direction the role points in
        :param neighbour_variable: the neighbour of the target concept
        :return:
        """
        print(query)
        roles_iterator = self._grakn_tx.query(query)

        def _roles_iterator():
            for answer in roles_iterator:
                # TODO See above, omitting due to bug
                # role_label = answer.get(role_variable).label()
                role_label = UNKNOWN_ROLE_TARGET_PLAYS_LABEL
                relationship_concept = answer.get(neighbour_variable)
                relationship_info = concept.build_concept_info(relationship_concept)

                yield {'role_label': role_label, 'role_direction': role_direction, 'neighbour_info': relationship_info}

        return _roles_iterator()
