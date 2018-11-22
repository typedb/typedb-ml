import copy
import typing as typ

import collections
import grakn
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python import debug as tf_debug

import kgcn.src.encoder.boolean as boolean
import kgcn.src.encoder.encode as encode
import kgcn.src.encoder.schema as schema
import kgcn.src.encoder.tf_hub as tf_hub
import kgcn.src.examples.animal_trade.persistence as persistence
import kgcn.src.models.learners as base
import kgcn.src.models.manager as manager
import kgcn.src.neighbourhood.data.executor as data_ex
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.schema.executor as schema_ex
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.neighbourhood.schema.traversal as trav
import kgcn.src.preprocess.date_to_unixtime as date
import kgcn.src.preprocess.preprocess as pp
import kgcn.src.preprocess.raw_array_builder as raw

flags = tf.app.flags
FLAGS = flags.FLAGS

# flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
# flags.DEFINE_integer('classes_length', 2, 'Number of classes')
# flags.DEFINE_integer('features_length', 9 + 20 + 128, 'Number of features after encoding')
# flags.DEFINE_integer('starting_concepts_features_length', 20 + 128,
#                      'Number of features after encoding for the nodes of interest, which excludes the features for '
#                      'role_type and role_direction')
# flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
# flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
#                                           'and the final length of the embeddings')
#
# flags.DEFINE_integer('max_training_steps', 100, 'Max number of gradient steps to take during gradient descent')
# flags.DEFINE_string('log_dir', './out', 'directory to use to store data from training')

NO_DATA_TYPE = ''  # TODO Pass this to traversal/executor


def main():
    # entity_query = "match $x isa person, has name 'Sundar Pichai'; get;"
    entity_query = "match $x isa company, has name 'Google'; get;"
    uri = "localhost:48555"
    keyspace = "test_schema"
    client = grakn.Grakn(uri=uri)
    session = client.session(keyspace=keyspace)
    tx = session.transaction(grakn.TxType.WRITE)

    neighbour_sample_sizes = (4, 3)

    sampling_method = ordered.ordered_sample

    samplers = []
    for sample_size in neighbour_sample_sizes:
        samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size * 2))

    # Strategies
    role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=False, include_metatypes=False)
    thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=False, include_metatypes=False)

    traversal_strategies = {'role': role_schema_strategy,
                            'thing': thing_schema_strategy}

    concepts = [concept.get('x') for concept in list(tx.query(entity_query))]

    kgcn = KGCN(tx, traversal_strategies, samplers)

    kgcn.train(tx, concepts, np.array([[1, 0]], dtype=np.float32))
    kgcn.predict(tx, concepts)


class KGCNFeature:
    def __init__(self, raw_data_type, preprocessor, tensorisor, encoder):
        self.raw_data_type = raw_data_type
        self.preprocessor = preprocessor
        self.tensorisor = tensorisor
        self.encoder = encoder


class KGCN:

    def __init__(self, schema_tx, traversal_strategies, traversal_samplers, features_to_exclude=(), storage_path=None):
        """
        A full Knowledge Graph Convolutional Network, running with TensorFlow and Grakn
        :param schema_tx:
        :return:
        """
        self._traversal_strategies = traversal_strategies
        self._traversal_samplers = traversal_samplers

        ################################################################################################################
        # Neighbour Traversals
        ################################################################################################################
        self._neighbour_sample_sizes = tuple(sampler.sample_size for sampler in self._traversal_samplers)

        ################################################################################################################
        # Schema Traversals
        ################################################################################################################

        # This depends upon the schema being the same for the keyspace used in training vs eval and predict
        schema_traversal_executor = schema_ex.TraversalExecutor(schema_tx)

        # THINGS
        thing_schema_traversal = trav.traverse_schema(self._traversal_strategies['thing'], schema_traversal_executor)

        # ROLES
        role_schema_traversal = trav.traverse_schema(self._traversal_strategies['role'], schema_traversal_executor)
        role_schema_traversal['has'] = ['has']

        ############################################################################################################
        # Encoders Initialisation
        ############################################################################################################

        with tf.name_scope('encoding_init') as scope:
            thing_encoder = schema.MultiHotSchemaTypeEncoder(thing_schema_traversal, name='thing_encoder')
            role_encoder = schema.MultiHotSchemaTypeEncoder(role_schema_traversal, name='role_encoder')

            # In case of issues https://github.com/tensorflow/hub/issues/61
            string_encoder = tf_hub.TensorFlowHubEncoder(
                "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1", 128)

            data_types = list(data_ex.DATA_TYPE_NAMES)
            data_types.insert(0, NO_DATA_TYPE)  # For the case where an entity or relationship is encountered
            data_types_traversal = {data_type: data_types for data_type in data_types}

            # Later a hierarchy could be added to data_type meaning. e.g. long and double are both numeric
            data_type_encoder = schema.MultiHotSchemaTypeEncoder(data_types_traversal, name='data_type_encoder')

        ################################################################################################################
        # Features
        ################################################################################################################
        F = KGCNFeature
        self._features = {
            'role_type': F(tf.string, lambda x: x, lambda x: tf.convert_to_tensor(x, dtype=tf.string), role_encoder),
            'role_direction': F(tf.int64, lambda x: x, lambda x: x, lambda x: tf.to_float(x, 'role_dir_to_float')),
            'neighbour_type': F(tf.string, lambda x: x, lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                                thing_encoder),
            'neighbour_data_type': F(tf.string, lambda x: x, lambda x: x,
                                     lambda x: data_type_encoder(tf.convert_to_tensor(x))),
            'neighbour_value_long': F(tf.int64, lambda x: x, lambda x: x,
                                      lambda x: tf.to_float(x, name='long_to_float')),
            'neighbour_value_double': F(tf.float32, lambda x: x, lambda x: x, lambda x: x),
            'neighbour_value_boolean': F(tf.int64, lambda x: x, lambda x: x, lambda x: tf.to_float(
                boolean.one_hot_boolean_encode(x, 'boolean_1_hot'), name='boolean_to_float')),
            'neighbour_value_date': F(tf.int64, date.datetime_to_unixtime, lambda x: x,
                                      lambda x: tf.nn.l2_normalize(tf.to_float(x, name='date_to_float'))),
            'neighbour_value_string': F(tf.string, lambda x: x, lambda x: x, string_encoder)}

        for feature_to_exclude in features_to_exclude:
            del self._features[feature_to_exclude]

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################
        self._raw_builder = raw.RawArrayBuilder(self._neighbour_sample_sizes)

        # Preprocessors
        self._preprocessors = {feature_name: feature.preprocessor for feature_name, feature in self._features.items()}

        ################################################################################################################
        # Placeholders
        ################################################################################################################

        self._feature_types = {feature_name: feature.raw_data_type for feature_name, feature in self._features.items()}

        self._all_feature_types = [copy.copy(self._feature_types) for _ in range(len(self._neighbour_sample_sizes) + 1)]
        # Remove role placeholders for the starting concepts (there are no roles for them)
        del self._all_feature_types[-1]['role_type']
        del self._all_feature_types[-1]['role_direction']

        # Build the placeholders for the neighbourhood_depths for each feature type
        self._raw_array_placeholders = build_array_placeholders(None, self._neighbour_sample_sizes, 1,
                                                                self._all_feature_types, name='array_input')

        # if labels is not None:
        # Build the placeholder for the labels
        self._labels_placeholder = manager.build_labels_placeholder(None, FLAGS.classes_length,
                                                                    name='labels_input')
        tf.summary.histogram('labels_input', self._labels_placeholder)

        # Saving of input values
        self._input_saver = tf.train.Saver()

        ################################################################################################################
        # Tensorising
        ################################################################################################################

        # Tensorisors
        self._tensorisors = {feature_name: feature.tensorisor for feature_name, feature in self._features.items()}

        ################################################################################################################
        # Tensorising
        ################################################################################################################

        # Any steps needed to get arrays ready for the rest of the pipeline
        with tf.name_scope('tensorising') as scope:
            tensorised_arrays = pp.preprocess_all(self._raw_array_placeholders, self._tensorisors)

        ################################################################################################################
        # Encoding
        ################################################################################################################
        self._encoders = {feature_name: feature.encoder for feature_name, feature in self._features.items()}
        encoded_arrays = encode.encode_all(tensorised_arrays, self._encoders)
        print('Encoded shapes')
        print([encoded_array.shape for encoded_array in encoded_arrays])

        ################################################################################################################
        # Learner
        ################################################################################################################

        # Create a session for running Ops on the Graph.
        self._sess = tf.Session()
        self._graph = tf.get_default_graph()
        if FLAGS.debug:
            self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)

        features_lengths = [FLAGS.features_length] * len(self._neighbour_sample_sizes)
        features_lengths[-1] = FLAGS.starting_concepts_features_length
        print(features_lengths)

        # optimizer = tf.train.AdagradOptimizer(learning_rate=FLAGS.learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        learner = base.SupervisedAccumulationLearner(FLAGS.classes_length, features_lengths,
                                                     FLAGS.aggregated_length,
                                                     FLAGS.output_length, self._neighbour_sample_sizes, optimizer,
                                                     sigmoid_loss=False,
                                                     regularisation_weight=0.0, classification_dropout_keep_prob=1.0,
                                                     classification_activation=lambda x: x,
                                                     # classification_activation=tf.nn.tanh,
                                                     # Moves to nn.math.tanh in r1.12
                                                     # classification_activation=tf.nn.sigmoid,
                                                     # classification_activation=tf.nn.softsign,
                                                     classification_regularizer=layers.l2_regularizer(scale=0.1),
                                                     classification_kernel_initializer=
                                                     tf.contrib.layers.xavier_initializer())

        self._learning_manager = manager.LearningManager(learner, FLAGS.max_training_steps, FLAGS.log_dir)
        self._learning_manager(self._sess, encoded_arrays, self._labels_placeholder)  # Build the graph

        ################################################################################################################
        # Calls to the Learning manager
        ################################################################################################################
        self._storage_path = storage_path
        self._mode_params = {
            tf.estimator.ModeKeys.TRAIN: KGCN.ModeParams(self._learning_manager.train, 'train.p'),
            tf.estimator.ModeKeys.EVAL: KGCN.ModeParams(self._learning_manager.evaluate, 'eval.p'),
            tf.estimator.ModeKeys.PREDICT: KGCN.ModeParams(self._learning_manager.predict, 'predict.p'),
        }

    def _build_feed_dict(self, tx, concepts, labels=None):
        ################################################################################################################
        # Neighbour Traversals
        ################################################################################################################
        concept_infos = [data_ex.build_concept_info(concept) for concept in concepts]

        data_executor = data_ex.TraversalExecutor(tx)
        neighourhood_traverser = trv.NeighbourhoodTraverser(data_executor, self._traversal_samplers)

        neighbourhood_depths = [neighourhood_traverser(concept_info) for concept_info in concept_infos]
        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################
        raw_array_depths = self._raw_builder.build_raw_arrays(neighbour_roles)
        raw_array_depths = pp.preprocess_all(raw_array_depths, self._preprocessors)

        ################################################################################################################
        # Feeding
        ################################################################################################################
        feed_dict = {}
        if labels is not None:
            feed_dict[self._labels_placeholder] = labels

        for raw_array_placeholder, raw_arrays_dict in zip(self._raw_array_placeholders, raw_array_depths):

            # for feature_type_name in set(self._features_to_include).intersection(set(raw_arrays_dict.keys())):
            for feature_type_name in set(self._features).intersection(set(raw_arrays_dict.keys())):
                feed_dict[raw_array_placeholder[feature_type_name]] = raw_arrays_dict[feature_type_name]

        return feed_dict

    class ModeParams:
        def __init__(self, func, file_suffix):
            self.func = func
            self.file_suffix = file_suffix

    def _pack_feed_dict(self, feed_dict):
        feed_dict_placeholder_names_as_keys = {}

        for placeholder, value in feed_dict.items():
            feed_dict_placeholder_names_as_keys[placeholder.name] = value

        return feed_dict_placeholder_names_as_keys

    def _unpack_feed_dict(self, packed_feed_dict):

        print('Unpacking')
        unpacked_feed_dict = {}
        for placeholder_name, value in packed_feed_dict.items():
            unpacked_feed_dict[self._graph.get_tensor_by_name(placeholder_name)] = value

        return unpacked_feed_dict

    def get_feed_dict(self, mode, tx=None, concepts=None, labels=None, load=False, save=True):

        file_path = self._storage_path + self._mode_params[mode].file_suffix

        if load:
            feed_dict = self._unpack_feed_dict(persistence.load_variable(file_path))
        else:
            feed_dict = self._build_feed_dict(tx, concepts, labels=labels)

            if save:
                if self._storage_path is None:
                    raise ValueError('Cannot save data without a path to save to')

                persistence.save_variable(self._pack_feed_dict(feed_dict), file_path)

                # self._savers[mode] = tf.train.Saver(feed_dict)
                # saver = tf.train.Saver(feed_dict)
                # saver.save(self._sess, file_path)
                # self._savers[mode].save(self._sess, file_path)

        return feed_dict

    def model_fn(self, mode, tx=None, concepts=None, labels=None, load=False, save=True):

        feed_dict = self.get_feed_dict(mode, tx, concepts, labels=labels, load=load, save=save)
        self._mode_params[mode].func(self._sess, feed_dict)

    def train(self, tx, concepts, labels):
        self.model_fn(tf.estimator.ModeKeys.TRAIN, tx, concepts, labels, save=True)

    def evaluate(self, tx, concepts, labels):
        self.model_fn(tf.estimator.ModeKeys.EVAL, tx, concepts, labels, save=True)

    def predict(self, tx, concepts):
        self.model_fn(tf.estimator.ModeKeys.PREDICT, tx, concepts)

    def train_from_file(self):
        self.model_fn(tf.estimator.ModeKeys.TRAIN, load=True)

    def evaluate_from_file(self):
        self.model_fn(tf.estimator.ModeKeys.EVAL, load=True)

    def predict_from_file(self):
        self.model_fn(tf.estimator.ModeKeys.PREDICT, load=True)


def build_array_placeholders(batch_size, neighbourhood_sizes, features_length,
                             feature_types: typ.Union[typ.List[typ.MutableMapping[str, tf.DType]], tf.DType],
                             name=None):
    array_neighbourhood_sizes = list(reversed(neighbourhood_sizes))
    neighbourhood_placeholders = []

    histogram_allowed_dtypes = [tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64, tf.bfloat16,
                                tf.uint16, tf.float16, tf.uint32, tf.uint64]

    for i in range(len(array_neighbourhood_sizes) + 1):
        shape = [batch_size] + list(array_neighbourhood_sizes[i:]) + [features_length]
        phs = {}
        for name, data_type in feature_types[i].items():
            phs[name] = tf.placeholder(data_type, shape=shape, name=name)
            if data_type in histogram_allowed_dtypes:
                tf.summary.histogram('input/' + name, phs[name])

        neighbourhood_placeholders.append(phs)
    return neighbourhood_placeholders


if __name__ == "__main__":
    main()
