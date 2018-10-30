import collections
import unittest

import numpy as np

import kgcn.src.preprocessing.encoders.encode as encode
import tensorflow as tf

import kgcn.src.preprocessing.to_array.raw_array_building as raw
import kgcn.src.preprocessing.preprocess as pp

import grakn

import kgcn.src.neighbourhood.schema.traversal as trav
import kgcn.src.neighbourhood.schema.executor as ex
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.preprocessing.encoders.schema as schema
import kgcn.src.preprocessing.encoders.boolean as boolean

tf.enable_eager_execution()


def build_encoders(keyspace, uri="localhost:48555"):
    client = grakn.Grakn(uri=uri)
    session = client.session(keyspace=keyspace)
    tx = session.transaction(grakn.TxType.WRITE)

    traversal_executor = ex.TraversalExecutor(tx)
    # ================ THINGS ======================
    thing_schema_strategy = schema_strat.SchemaTraversalStrategy(kind="thing", include_implicit=True,
                                                                 include_metatypes=False)

    thing_schema_traversal = trav.traverse_schema(thing_schema_strategy, traversal_executor)
    thing_encoder = schema.MultiHotSchemaTypeEncoder(thing_schema_traversal)

    # ================ ROLES ======================
    role_schema_strategy = schema_strat.SchemaTraversalStrategy(kind="role", include_implicit=True,
                                                                include_metatypes=False)
    role_schema_traversal = trav.traverse_schema(role_schema_strategy, traversal_executor)
    role_encoder = schema.MultiHotSchemaTypeEncoder(role_schema_traversal)

    encoders = {'role_type': role_encoder,
                'role_direction': lambda x: x,
                'neighbour_type': thing_encoder,
                'neighbour_data_type': lambda x: x,
                'neighbour_value_long': lambda x: x,
                'neighbour_value_double': lambda x: x,
                'neighbour_value_boolean': lambda x: tf.cast(boolean.one_hot_boolean_encode(x), dtype=tf.float64),  # TODO Hacky, don't like it
                'neighbour_value_date': lambda x: x,
                'neighbour_value_string': lambda x: x}

    return encoders


class TestEncode(unittest.TestCase):

    def test_input_output(self):
        encoders = build_encoders(keyspace="test_schema", uri="localhost:48555")

        array_data_types = collections.OrderedDict(
            [('role_type', ('U25', 'employee')), ('role_direction', (np.int, 0)), ('neighbour_type', ('U25', 'person')),
             ('neighbour_data_type', (np.int, -1)), ('neighbour_value_long', (np.int, 0)),
             ('neighbour_value_double', (np.float, 0.0)), ('neighbour_value_boolean', (np.int, -1)),
             ('neighbour_value_date', (np.int, 0)), ('neighbour_value_string', (np.float, 0.0))])

        example_arrays = raw.build_default_arrays((3, 2), 4, array_data_types)
        print(example_arrays)

        preprocessed_example_arrays = pp.preprocess(example_arrays)
        print(encode.encode(preprocessed_example_arrays, encoders))
