import grakn

GET_THING_TYPES_QUERY = "match $x sub thing; get;"
GET_ROLE_TYPES_QUERY = "match $x sub role; get;"

METATYPE_LABELS = ['thing', 'entity', 'relationship', 'attribute', 'role', '@has-attribute-value',
                   '@has-attribute-owner', '@has-attribute']


def labels_from_types(schema_concept_types):
    for type in schema_concept_types:
        yield type.label()


def get_sups_labels_per_type(schema_concept_types, include_metatypes=False, include_self=False):
    for schema_concept_type in schema_concept_types:
        super_types = schema_concept_type.sups()

        supertype_labels = []
        for super_type in super_types:
            super_type_label = super_type.label()

            if not (((not include_self) and super_type_label == schema_concept_type.label()) or (
                (not include_metatypes) and super_type.label() in METATYPE_LABELS)):
                supertype_labels.append(super_type.label())
        yield schema_concept_type.label(), supertype_labels


class TraversalExecutor:
    def __init__(self, grakn_tx):
        self._grakn_tx = grakn_tx

    def get_schema_concept_types(self, get_types_query, include_implicit=True, include_metatypes=False):

        for answer in self._grakn_tx.query(get_types_query):
            t = answer.get('x')

            if not (((not include_implicit) and t.is_implicit()) or (
                    (not include_metatypes) and t.label() in METATYPE_LABELS)):
                yield t


if __name__ == '__main__':
    client = grakn.Grakn(uri="localhost:48555")
    session = client.session(keyspace="test_schema")
    tx = session.transaction(grakn.TxType.WRITE)

    print("================= THINGS ======================")
    te = TraversalExecutor(tx)
    schema_concept_types = te.get_schema_concept_types(GET_THING_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    labels = labels_from_types(schema_concept_types)
    print(list(labels))

    schema_concept_types = te.get_schema_concept_types(GET_THING_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    super_types = get_sups_labels_per_type(schema_concept_types, include_self=True, include_metatypes=False)
    print("==== super types ====")
    [print(super_type) for super_type in super_types]

    print("================= ROLES ======================")
    schema_concept_types = te.get_schema_concept_types(GET_ROLE_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    labels = labels_from_types(schema_concept_types)
    print(list(labels))

    schema_concept_types = te.get_schema_concept_types(GET_ROLE_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    super_types = get_sups_labels_per_type(schema_concept_types, include_self=True, include_metatypes=False)
    print("==== super types ====")
    [print(super_type) for super_type in super_types]


