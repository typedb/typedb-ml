import kgcn.src.neighbourhood.data.concept as concept
import kgcn.src.neighbourhood.data.strategy as strat
import kgcn.src.neighbourhood.data.traversal as trv


def gen(elements):
    for el in elements:
        yield el


def mock_traversal_output():
    c = trv.ConceptInfoWithNeighbourhood(
        concept.ConceptInfo("0", "person", "entity"),
        gen([
            trv.NeighbourRole("employee", strat.TARGET_PLAYS, trv.ConceptInfoWithNeighbourhood(
                concept.ConceptInfo("1", "employment", "relationship"),
                gen([
                    trv.NeighbourRole("employer", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("2", "company", "entity"), gen([])
                    )),
                    trv.NeighbourRole("employer", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("2", "company", "entity"), gen([])
                    )),
                    trv.NeighbourRole("employer", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("2", "company", "entity"), gen([])
                    ))
                ])
            )),
            trv.NeighbourRole("@has-name-owner", strat.TARGET_PLAYS, trv.ConceptInfoWithNeighbourhood(
                concept.ConceptInfo("3", "@has-name", "relationship"),
                gen([
                    trv.NeighbourRole("@has-name-value", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    )),
                    trv.NeighbourRole("@has-name-value", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    )),
                    trv.NeighbourRole("@has-name-value", strat.NEIGHBOUR_PLAYS, trv.ConceptInfoWithNeighbourhood(
                        concept.ConceptInfo("4", "name", "attribute", data_type='string', value="Employee Name"),
                        gen([])
                    ))
                ])
            ))

        ]))
    return c
