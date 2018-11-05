import kgcn.src.neighbourhood.data.executor as ex
import kgcn.src.neighbourhood.data.traversal as trv


def gen(elements):
    for el in elements:
        yield el


def mock_traversal_output():
    c = trv.ConceptInfoWithNeighbourhood(
        ex.ConceptInfo("0", "person", "entity"),
        gen([
            trv.NeighbourRole("employee", ex.TARGET_PLAYS, trv.ConceptInfoWithNeighbourhood(
                ex.ConceptInfo("1", "employment", "relationship"),
                gen([
                    trv.NeighbourRole("employer", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("2", "company", "entity"), gen([])
                    )),
                    trv.NeighbourRole("employer", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("2", "company", "entity"), gen([])
                    )),
                    trv.NeighbourRole("employer", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("2", "company", "entity"), gen([])
                    ))
                ])
            )),
            trv.NeighbourRole("@has-name-owner", ex.TARGET_PLAYS, trv.ConceptInfoWithNeighbourhood(
                ex.ConceptInfo("3", "@has-name", "relationship"),
                gen([
                    trv.NeighbourRole("@has-name-value", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    )),
                    trv.NeighbourRole("@has-name-value", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    )),
                    trv.NeighbourRole("@has-name-value", ex.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        ex.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    ))
                ])
            ))

        ]))
    return c


def _build_data(role_label, role_direction, neighbour_id, neighbour_type, neighbour_metatype, data_type=None,
                value=None):
    return {'role_label': role_label, 'role_direction': role_direction,
            'neighbour_info': ex.ConceptInfo(neighbour_id, neighbour_type, neighbour_metatype, data_type=data_type,
                                             value=value)}


def _role_wrapper(outcome, role_direction, query_direction):
    if role_direction == query_direction:
        return gen(outcome)
    else:
        return gen([])


def mock_executor(query_direction, *args):

    concept_id = args[0]
    if concept_id == "0":

        role_direction = ex.TARGET_PLAYS
        return _role_wrapper([
                _build_data("employee", role_direction, "1", "employment", "relationship"),
                _build_data("@has-name-owner", role_direction, "3", "@has-name", "relationship")
            ],
            role_direction,
            query_direction
        )

    elif concept_id == "1":

        role_direction = ex.NEIGHBOUR_PLAYS
        return _role_wrapper([_build_data("employer", role_direction, "2", "company", "entity")]*3,
                             role_direction,
                             query_direction)

    elif concept_id == "3":

        role_direction = ex.NEIGHBOUR_PLAYS
        return _role_wrapper([_build_data("@has-name-value", role_direction, "4", "name", "attribute",
                                          data_type='string', value="Employee Name")] * 3,
                             role_direction,
                             query_direction)
    else:
        raise ValueError("This concept id hasn't been mocked")
