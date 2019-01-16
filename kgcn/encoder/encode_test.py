import unittest

import grakn
import numpy as np
import tensorflow as tf

import kgcn.encoder.encode as encode


class TestEncode(unittest.TestCase):

    def test_encode(self):
        keyspace="test_schema"
        uri="localhost:48555"
        client = grakn.Grakn(uri=uri)
        session = client.session(keyspace=keyspace)
        tx = session.transaction(grakn.TxType.WRITE)
        encoder = encode.Encoder(tx, include_implicit=False, include_metatypes=False)

        placeholders = [
            {'role_type': tf.placeholder(dtype=tf.string, shape=(None, 1)),
             'role_direction': tf.placeholder(dtype=tf.int64, shape=(None, 1)),
             'neighbour_type': tf.placeholder(dtype=tf.string, shape=(None, 1)),
             'neighbour_data_type': tf.placeholder(dtype=tf.string, shape=(None, 1)),
             'neighbour_value_long': tf.placeholder(dtype=tf.int64, shape=(None, 1)),
             'neighbour_value_double': tf.placeholder(dtype=tf.float32, shape=(None, 1)),
             'neighbour_value_boolean': tf.placeholder(dtype=tf.int64, shape=(None, 1)),
             'neighbour_value_date': tf.placeholder(dtype=tf.int64, shape=(None, 1)),
             'neighbour_value_string': tf.placeholder(dtype=tf.string, shape=(None, 1))
             }
        ]

        encoded_output = encoder(placeholders)

        example_arrays = {
            'role_type': np.full((4, 1), fill_value='employee', dtype=np.dtype('U50')),
            'role_direction': np.full((4, 1), fill_value=0, dtype=np.int),
            'neighbour_type': np.full((4, 1), fill_value='person', dtype=np.dtype('U50')),
            'neighbour_data_type': np.full((4, 1), fill_value='', dtype=np.dtype('U10')),
            'neighbour_value_long': np.full((4, 1), fill_value=0, dtype=np.int),
            'neighbour_value_double': np.full((4, 1), fill_value=0.0, dtype=np.float),
            'neighbour_value_boolean': np.full((4, 1), fill_value=0, dtype=np.int),
            'neighbour_value_date': np.full((4, 1), fill_value=0, dtype=np.int),
            'neighbour_value_string': np.full((4, 1), fill_value='', dtype=np.dtype('U50'))
        }

        feed_dict = {placeholder: example_arrays[placeholder_name] for placeholder_name, placeholder in
                     placeholders[0].items()}

        init_global = tf.global_variables_initializer()
        init_tables = tf.tables_initializer()

        tf_session = tf.Session()
        tf_session.run(init_global)
        tf_session.run(init_tables)

        tf_session.run(encoded_output, feed_dict=feed_dict)


if __name__ == "__main__":
    unittest.main()
