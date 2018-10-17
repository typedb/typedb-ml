import tensorflow as tf


def indices_from_categories(feature_strings, mapping_strings):
    # Remember to use tf.tables_initializer().run()
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=-1)
    indices = table.lookup(feature_strings)
    return indices  # int64 IDs
