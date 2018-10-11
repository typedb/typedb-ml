import collections

import tensorflow as tf
import numpy as np

# (num starting_nodes (num 'neighbours' at depth 0), num neighbours at depth 1, num neighbours at depth 2 ...)
traversal_dims = (1, 2)  # TODO should this be a constant tensor?
# traversal_dims = tf.constant([1, 2, 2])

TARGET_NODE_FEATURE_LENGTH = None
AGGREGATED_FEATURE_LENGTH = 64
STRING_DELIMITER = ' '

# def put_in_array():
#     pass
#
#
# numpy_arrays = [{key: np.empty(shape, dtype=dtype) for key, dtype in data_types.items()} for shape in [1, (1, 3)]]
#
# for target_with_neighbours in basic_data:
#     target = target_with_neighbours[0]
#     d = numpy_arrays[0]
#     current_indices = indices +
#     for i, key in enumerate(list(data_types.keys())):
#         d[key][current_indices] =

basic_data = [
    (("", 0, "person", 0, 0, 0, 0, 0, ""),
     [("has", -1, "name",       5, 0,   0, 0, 0, "Employee Name"),
      ("has", -1, "age",        1, 32,  0, 0, 0, ""),
      ("wife", 1, "marriage",   0, 0,   0, 0, 0, "")])
]

basic_data_arrays = [
    {
        'role_type': np.array([""]),
        'role_direction': np.array([0]),
        'neighbour_type': np.array(["person"]),
        'neighbour_data_type': np.array([0]),
        'neighbour_value_long': np.array([0]),
        'neighbour_value_double': np.array([0]),
        'neighbour_value_boolean': np.array([0]),
        'neighbour_value_date': np.array([0]),
        'neighbour_value_string': np.array([""]),
    },
    {
        'role_type': np.array([["has", "has", "wife"]]),
        'role_direction': np.array([[-1, -1, 1]]),
        'neighbour_type': np.array([["name", "age", "marriage"]]),
        'neighbour_data_type': np.array([[5, 1, 0]]),
        'neighbour_value_long': np.array([[0, 32, 0]]),
        'neighbour_value_double': np.array([[0, 0, 0]]),
        'neighbour_value_boolean': np.array([[0, 0, 0]]),
        'neighbour_value_date': np.array([[0, 0, 0]]),
        'neighbour_value_string': np.array([["Employee Name", "", ""]]),
    }
]

thing_types = ['person', 'name', 'age', 'marriage']
role_types = ['', 'has', 'wife']

data_types = collections.OrderedDict(
    [
        ('role_type', 'U25'),
        ('role_direction', np.int8),
        ('neighbour_type', 'U25'),
        ('neighbour_data_type', np.int8),
        ('neighbour_value_long', np.int32),  # 1
        ('neighbour_value_double', np.float32),  # 2
        ('neighbour_value_boolean', np.int8),  # 3
        ('neighbour_value_date', np.int64),  # 4  # Just use unix time for simplicity
        ('neighbour_value_string', 'U25')  # 5
    ])

tf_data_types = (tf.string, tf.int8, tf.string, tf.int8, tf.int32, tf.float32, tf.int8, tf.int64, tf.string)

EAGER = True

if EAGER:
    tf.enable_eager_execution()
else:
    pass

# raw_features = []
# for i in range(len(traversal_dims)):
#
#     dims_at_this_depth = traversal_dims[:i+1]
#     raw_features.append(np.ones(dims_at_this_depth, dtype=[(key, value) for key, value in data_types.items()]))


# def gen(collection):
#     for item in collection:
#         yield item


# tf.FixedLenFeature
# try https://www.tensorflow.org/api_docs/python/tf/name_scope
# tf.convert_to_tensor

datasets = []
for depth in range(len(traversal_dims)):
    dims_at_this_depth = traversal_dims[:depth + 1]
    dims_at_this_depth_with_features = tuple([len(tf_data_types)] + list(dims_at_this_depth))

    if EAGER:
        dict_to_list = (basic_data_arrays[depth][name] for name in list(data_types.keys()))
        datasets.append(tf.data.Dataset.from_generator(lambda: iter(dict_to_list), tf_data_types, dims_at_this_depth_with_features))
    else:
        # dataset_format = {key: tf.placeholder(dtype=data_type, shape=dims_at_this_depth) for key, (value, data_type)
        #                   in data_types.items()}
        # input_tensors =
        # datasets.append(tf.data.Dataset.from_tensor_slices(input_tensors))
        pass

# dataset = tf.data.Dataset.zip(tuple(datasets))

# tf.string_split(neighbour_value_string)
# tf.unique()


# print("basic_data_arrays")
# print(basic_data_arrays)

print("datasets")
print(datasets)

# iterator = dataset.make_one_shot_iterator()
iterator = datasets[0].make_one_shot_iterator()

for data in iterator:
    print('================================')
    print(data)


###############################################################
# Encoding
# One-hot encode role_type, neighbour_type
# Encode neighbour_value_date
# Encode neighbour_value_string
###############################################################












# writer = tf.summary.FileWriter('out/.')
# writer.add_graph(tf.get_default_graph())

# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)

# print(sess.run(y, {x: [[1, 2, 3],[4, 5, 6]]}))
# print(sess.run(z, feed_dict={x: 3, y: 4.5}))
# print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

# if __name__ == "__main__":
#     pass
