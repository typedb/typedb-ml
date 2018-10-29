

class SchemaTraversalStrategy:
    """
    Class to contain all of the details of how the knowledge graph traversals are conducted
    """
    def __init__(self, kind, include_implicit=True, include_metatypes=False):
        if kind not in ("thing", "role"):
            raise ValueError("Schema traversals must traverse either all things or all roles. Provide one of these.")
        self.kind = kind  # either traverse things or roles
        self.include_implicit = include_implicit
        self.include_metatypes = include_metatypes
