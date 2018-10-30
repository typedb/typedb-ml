import abc

GET_THING_TYPES_QUERY = "match $x sub thing; get;"
GET_ROLE_TYPES_QUERY = "match $x sub role; get;"


class SchemaTraversalStrategy(abc.ABC):
    QUERY = None

    """
    Class to contain all of the details of how the knowledge graph traversals are conducted
    """
    def __init__(self, include_implicit=True, include_metatypes=False):
        self.include_implicit = include_implicit
        self.include_metatypes = include_metatypes

    @property
    def query(self):
        if self.QUERY is None:
            raise NotImplementedError('The query to be performed needs to be defined')
        return self.QUERY


class SchemaRoleTraversalStrategy(SchemaTraversalStrategy):
    QUERY = GET_ROLE_TYPES_QUERY


class SchemaThingTraversalStrategy(SchemaTraversalStrategy):
    QUERY = GET_THING_TYPES_QUERY
