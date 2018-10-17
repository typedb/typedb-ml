import numpy as np


def build_adjacency_matrix(schema_concept_super_types):
    adj = np.zeros((len(schema_concept_super_types), len(schema_concept_super_types)))
    schema_concepts = list(schema_concept_super_types.keys())
    for schema_concept_index, (schema_concept, super_types) in enumerate(schema_concept_super_types.items()):
        for super_type in super_types:
            super_type_index = schema_concepts.index(super_type)
            adj[schema_concept_index, super_type_index] = 1
    return schema_concepts, adj
