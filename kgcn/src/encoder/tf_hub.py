import tensorflow_hub as hub
import tensorflow as tf


class TensorFlowHubEncoder:

    def __init__(self, module_url, name='tf_hub_encoder'):
        self._embed = hub.Module(module_url)
        self._name = name

    def __call__(self, features: tf.Tensor):
        with tf.name_scope(name=self._name) as scope:
            shape = list(features.shape)
            print(shape)
            flattened_features = tf.reshape(features, [-1])
            flat_embeddings = self._embed(flattened_features)
            shape[-1] = -1
            embeddings = tf.reshape(flat_embeddings, shape)
            return embeddings

