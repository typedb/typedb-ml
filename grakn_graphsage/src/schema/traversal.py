import grakn


GET_THING_TYPES_QUERY = "match $x sub thing; get;"
GET_ROLE_TYPES_QUERY = "match $x sub role; get;"

METATYPE_LABELS = ['thing', 'entity', 'relationship', 'attribute', 'role', '@has-attribute-value',
                   '@has-attribute-owner', '@has-attribute']


def labels_from_types(schema_concept_types):
    for type in schema_concept_types:
        yield type.label()


def get_schema_concept_types(grakn_tx, get_types_query, include_implicit=True, include_metatypes=False):

    schema_concepts = []

    for answer in grakn_tx.query(get_types_query):
        t = answer.get('x')

        if not (((not include_implicit) and t.is_implicit()) or (
                (not include_metatypes) and t.label() in METATYPE_LABELS)):
            schema_concepts.append(t.label())
            yield t


if __name__ == '__main__':
    client = grakn.Grakn(uri="localhost:48555")
    session = client.session(keyspace="test_schema")
    tx = session.transaction(grakn.TxType.WRITE)

    schema_concept_types = get_schema_concept_types(tx, GET_THING_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    labels = labels_from_types(schema_concept_types)
    print(list(labels))

    print("=======================================")
    schema_concept_types = get_schema_concept_types(tx, GET_ROLE_TYPES_QUERY, include_implicit=True, include_metatypes=False)
    labels = labels_from_types(schema_concept_types)
    print(list(labels))
