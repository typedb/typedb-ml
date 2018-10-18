import numpy as np
import tensorflow as tf

import kgcn.src.models.base as models

tf.enable_eager_execution()

num_samples = 30
neighbourhood_sizes = (4, 3)
feature_length = 8
neighbourhood_shape = list(neighbourhood_sizes) + [feature_length]
shapes = [[num_samples] + neighbourhood_shape[i:] for i in range(len(neighbourhood_shape))]


raw_neighbourhood = [np.ones(shape) for shape in shapes]

neighbourhood = []
for n in raw_neighbourhood:
    neighbourhood.append(tf.convert_to_tensor(n, dtype=tf.float64))

print([n.shape for n in neighbourhood])

raw_labels = [1] * num_samples
labels = tf.convert_to_tensor(raw_labels, dtype=tf.float64)
print(labels.shape)

aggregated_length = 12
output_length = 17

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

model = models.SupervisedModel(len(raw_labels), feature_length, aggregated_length, output_length=output_length,
                               neighbourhood_sizes=neighbourhood_sizes, optimizer=optimizer)

embedding = model.embedding(neighbourhood)
print("==============================")
print(embedding)

print("===== Loss =====")
raw_predictions = [1] * num_samples
predictions = tf.convert_to_tensor(raw_predictions, dtype=tf.float64)
loss = model.loss(predictions=predictions, labels=labels)

print(loss)
