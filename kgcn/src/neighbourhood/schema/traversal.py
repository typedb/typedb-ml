import collections

METATYPE_LABELS = ['thing', 'entity', 'relationship', 'attribute', 'role', '@has-attribute-value',
                   '@has-attribute-owner', '@has-attribute']


def labels_from_types(schema_concept_types):
    for type in schema_concept_types:
        yield type.label()


def get_sups_labels_per_type(schema_concept_types, include_metatypes=False, include_self=False):

    schema_concept_super_types = collections.OrderedDict()

    for schema_concept_type in schema_concept_types:
        super_types = schema_concept_type.sups()

        super_type_labels = []
        for super_type in super_types:
            super_type_label = super_type.label()

            if not (((not include_self) and super_type_label == schema_concept_type.label()) or (
                (not include_metatypes) and super_type.label() in METATYPE_LABELS)):
                super_type_labels.append(super_type.label())
        schema_concept_super_types[schema_concept_type.label()] = super_type_labels
    return schema_concept_super_types


def traverse_schema(schema_strategy, traversal_executor):
    schema_concept_types = \
        list(traversal_executor.get_schema_concept_types(schema_strategy.query,
                                                         include_implicit=schema_strategy.include_implicit,
                                                         include_metatypes=schema_strategy.include_metatypes))

    schema_concept_super_type_labels = get_sups_labels_per_type(schema_concept_types, include_self=True,
                                                                include_metatypes=schema_strategy.include_metatypes)

    return schema_concept_super_type_labels
