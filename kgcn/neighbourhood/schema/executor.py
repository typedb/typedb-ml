import kgcn.neighbourhood.schema.traversal as trav


class TraversalExecutor:
    def __init__(self, grakn_tx):
        self._grakn_tx = grakn_tx

    def get_schema_concept_types(self, get_types_query, include_implicit=True, include_metatypes=False):

        for answer in self._grakn_tx.query(get_types_query):
            t = answer.get('x')

            if not (((not include_implicit) and t.is_implicit()) or (
                    (not include_metatypes) and t.label() in trav.METATYPE_LABELS)):
                yield t
