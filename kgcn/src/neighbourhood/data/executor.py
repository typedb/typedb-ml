import kgcn.src.neighbourhood.data.concept as concept

UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL = "UNKNOWN_ROLE_NEIGHBOUR_PLAYS"
UNKNOWN_ROLE_TARGET_PLAYS_LABEL = "UNKNOWN_ROLE_TARGET_PLAYS"

TARGET_PLAYS = 0  # In this case, the neighbour is a relationship in which this concept plays a role
NEIGHBOUR_PLAYS = 1  # In this case the target


class TraversalExecutor:
    def __init__(self, grakn_tx):
        self._grakn_tx = grakn_tx

    def get_roles_played(self, target_concept_info, limit):

        # TODO Can't do this presently since querying for the role throws an exception
        # roles_played_query = (
        #     f"match $x id {target_concept_info.id}; $relationship($role: $x); offset {0}; limit {limit}; get
        # $relationship, "
        #     f"$role;")

        roles_played_query = (
            f"match $x id {target_concept_info.id}; $relationship($x); offset {0}; limit {limit}; get $relationship;")

        print(roles_played_query)
        roles_played_iterator = self._grakn_tx.query(roles_played_query)

        def _roles_played_iterator():
            for answer in roles_played_iterator:

                # TODO See above, omitting due to bug
                # role_concept = answer.get("role")
                role_label = UNKNOWN_ROLE_TARGET_PLAYS_LABEL
                relationship_concept = answer.get("relationship")
                relationship_info = concept.build_concept_info(relationship_concept)

                yield {'role_label': role_label, 'role_direction': TARGET_PLAYS, 'neighbour_info': relationship_info}

        return _roles_played_iterator()

    def get_roleplayers(self, target_concept_info, limit):
        # id and the concept type should be known (providing the concept type speeds up the query, which it shouldn't
        # since we provide the concept's id)

        # TODO Can't do this presently since querying for the role throws an exception
        # roleplayers_query = (
        #     f"match $relationship id {target_concept_info.id}; $relationship($role: $x) isa {
        # target_concept_info.type_label};
        # offset {0}; limit {limit}; get $x, $role;")

        roleplayers_query = (
            f"match $relationship id {target_concept_info.id}; $relationship($x) isa {target_concept_info.type_label}; "
            f"offset {0}; limit {limit}; get $x;")
        print(roleplayers_query)
        roleplayers_iterator = self._grakn_tx.query(roleplayers_query)

        def _get_roleplayers_iterator():
            for answer in roleplayers_iterator:
                # role_concept = answer.get("role")
                role_label = UNKNOWN_ROLE_NEIGHBOUR_PLAYS_LABEL
                roleplayer_concept = answer.get("x")
                roleplayer_info = concept.build_concept_info(roleplayer_concept)
                yield {'role_label': role_label, 'role_direction': NEIGHBOUR_PLAYS, 'neighbour_info': roleplayer_info}

        return _get_roleplayers_iterator()
