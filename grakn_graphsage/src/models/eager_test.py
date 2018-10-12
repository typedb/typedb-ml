import numpy as np
import tensorflow as tf

import grakn_graphsage.src.models.base as models

tf.enable_eager_execution()

neighbourhood_sizes = (4, 3)
feature_length = 8
neighbourhood_shape = list(neighbourhood_sizes) + [feature_length]
raw_neighbourhood = [np.ones(neighbourhood_shape[i:]) for i in range(len(neighbourhood_shape))]

neighbourhood = []
for n in raw_neighbourhood:
    neighbourhood.append(tf.convert_to_tensor(n, dtype=tf.float64))

neighbourhood[-1] = tf.expand_dims(neighbourhood[-1], axis=0)

print(neighbourhood)

raw_labels = [1]
labels = tf.convert_to_tensor(raw_labels, dtype=tf.float64)
print(labels)

aggregated_length = 12
output_length = 17

model = models.SupervisedModel(len(raw_labels), feature_length, aggregated_length, output_length=output_length,
                               neighbourhood_sizes=neighbourhood_sizes)

embedding = model.embedding(neighbourhood)
print("==============================")
print(embedding)
