import unittest

import collections
import grakn
import numpy as np
import tensorflow as tf

import kgcn.neighbourhood.schema.executor as ex
import kgcn.neighbourhood.schema.strategy as schema_strat
import kgcn.neighbourhood.schema.traversal as trav
import kgcn.encoder.boolean as boolean
import kgcn.encoder.encode as encode
import kgcn.encoder.schema as schema
import kgcn.preprocess.preprocess as pp
import kgcn.preprocess.raw_array_builder as raw
import kgcn.preprocess.date_to_unixtime as date

tf.enable_eager_execution()


def build_encoders(keyspace, uri="localhost:48555"):
    client = grakn.Grakn(uri=uri)
    session = client.session(keyspace=keyspace)
    tx = session.transaction(grakn.TxType.WRITE)

    traversal_executor = ex.TraversalExecutor(tx)
    # ================ THINGS ======================
    thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=True,
                                                                      include_metatypes=False)

    thing_schema_traversal = trav.traverse_schema(traversal_executor, query, include_implicit, include_metatypes)
    thing_encoder = schema.MultiHotSchemaTypeEncoder(thing_schema_traversal)

    # ================ ROLES ======================
    role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=True, include_metatypes=False)
    role_schema_traversal = trav.traverse_schema(traversal_executor, query, include_implicit, include_metatypes)
    role_encoder = schema.MultiHotSchemaTypeEncoder(role_schema_traversal)

    encoders = {'role_type': role_encoder,
                'role_direction': lambda x: x,
                'neighbour_type': thing_encoder,
                'neighbour_data_type': lambda x: x,
                'neighbour_value_long': lambda x: x,
                'neighbour_value_double': lambda x: x,
                'neighbour_value_boolean': lambda x: tf.to_double(boolean.one_hot_boolean_encode(x)),
                'neighbour_value_date': lambda x: x,
                'neighbour_value_string': lambda x: x}

    return encoders


class TestEncode(unittest.TestCase):

    def test_encode(self):
        encoders = build_encoders(keyspace="test_schema", uri="localhost:48555")

        array_data_types = collections.OrderedDict(
            [('role_type', ('U25', 'employee')), ('role_direction', (np.int, 0)), ('neighbour_type', ('U25', 'person')),
             ('neighbour_data_type', (np.int, -1)), ('neighbour_value_long', (np.int, 0)),
             ('neighbour_value_double', (np.float, 0.0)), ('neighbour_value_boolean', (np.int, -1)),
             ('neighbour_value_date', (np.int, 0)), ('neighbour_value_string', (np.float, 0.0))])

        example_arrays = raw.build_default_arrays((3, 2), 4, array_data_types)
        print(example_arrays)

        preprocessors = {'role_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                         'role_direction': lambda x: x,
                         'neighbour_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                         'neighbour_data_type': lambda x: x,
                         'neighbour_value_long': lambda x: x,
                         'neighbour_value_double': lambda x: x,
                         'neighbour_value_boolean': lambda x: x,
                         'neighbour_value_date': date.datetime_to_unixtime,
                         'neighbour_value_string': lambda x: x}

        preprocessed_example_arrays = pp.apply_operations(example_arrays, preprocessors)
        print(encode.encode_all(preprocessed_example_arrays, encoders))
