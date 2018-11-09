import collections

import grakn
import tensorflow as tf

import kgcn.src.encoder.boolean as boolean
import kgcn.src.encoder.encode as encode
import kgcn.src.encoder.schema as schema
import kgcn.src.encoder.tf_hub as tf_hub
import kgcn.src.models.training as training
import kgcn.src.neighbourhood.data.executor as data_ex
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.schema.executor as schema_ex
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.neighbourhood.schema.traversal as trav
import kgcn.src.preprocess.date_to_unixtime as date
import kgcn.src.preprocess.preprocess as pp
import kgcn.src.preprocess.raw_array_builder as raw
import kgcn.src.neighbourhood.data.sampling.ordered as ordered


flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('training_batch_size', 1, 'Training batch size')
flags.DEFINE_integer('neighbourhood_size_depth_1', 3, 'Neighbourhood size for depth 1')
flags.DEFINE_integer('neighbourhood_size_depth_2', 4, 'Neighbourhood size for depth 2')

flags.DEFINE_integer('classes_length', 2, 'Number of classes')
flags.DEFINE_integer('features_length', 128+30, 'Number of features after encoding')
flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
                                          'and the final length of the embeddings')

flags.DEFINE_integer('max_training_steps', 100, 'Max number of gradient steps to take during gradient descent')
flags.DEFINE_string('log_dir', './out', 'directory to use to store data from training')

NO_DATA_TYPE = ''  # TODO Pass this to traversal/executor
NEIGHBOURHOOD_SIZES = (FLAGS.neighbourhood_size_depth_2, FLAGS.neighbourhood_size_depth_1)


def main():
    # tf.enable_eager_execution()
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
    role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=True, include_metatypes=False)
    thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=True, include_metatypes=False)

    traversal_strategies = {'role': role_schema_strategy,
                            'thing': thing_schema_strategy}

    concepts = [concept.get('x') for concept in list(tx.query(entity_query))]

    kgcn = KGCN(tx, traversal_strategies, samplers)

    kgcn.model_fn(concepts, [[1, 0]])


class KGCN:

    def __init__(self, tx, traversal_strategies, traversal_samplers):
        self._tx = tx
        self._traversal_strategies = traversal_strategies
        self._traversal_samplers = traversal_samplers

    def model_fn(self, concepts, labels=None):
        """
        A full Knowledge Graph Convolutional Network, running with TensorFlow and Grakn
        :return:
        """

        concept_infos = [data_ex.build_concept_info(concept) for concept in concepts]

        data_executor = data_ex.TraversalExecutor(self._tx)

        neighourhood_traverser = trv.NeighbourhoodTraverser(data_executor, self._traversal_samplers)

        neighbourhood_depths = [neighourhood_traverser(concept_info) for concept_info in concept_infos]

        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################
        neighbour_sample_sizes = tuple(sampler.sample_size for sampler in self._traversal_samplers)
        raw_builder = raw.RawArrayBuilder(neighbour_sample_sizes, len(concepts))
        raw_arrays = raw_builder.build_raw_arrays(neighbour_roles)

        # with tf.name_scope('preprocessing') as scope:
        # Preprocessors
        preprocessors = {'role_type': lambda x: x,
                         'role_direction': lambda x: x,
                         'neighbour_type': lambda x: x,
                         'neighbour_data_type': lambda x: x,
                         'neighbour_value_long': lambda x: x,
                         'neighbour_value_double': lambda x: x,
                         'neighbour_value_boolean': lambda x: x,
                         'neighbour_value_date': date.datetime_to_unixtime,
                         'neighbour_value_string': lambda x: x}

        raw_arrays = pp.preprocess_all(raw_arrays, preprocessors)

        ################################################################################################################
        # Placeholders
        ################################################################################################################

        feature_types = collections.OrderedDict(
            [('role_type', tf.string),
             ('role_direction', tf.int64),
             ('neighbour_type', tf.string),
             ('neighbour_data_type', tf.string),
             ('neighbour_value_long', tf.int64),
             ('neighbour_value_double', tf.float32),
             ('neighbour_value_boolean', tf.int64),
             ('neighbour_value_date', tf.int64),
             ('neighbour_value_string', tf.string)])

        all_feature_types = [feature_types for _ in range(len(neighbour_sample_sizes) + 1)]
        # Remove role placeholders for the starting concepts (there are no roles for them)
        del all_feature_types[0]['role_type']
        del all_feature_types[0]['role_direction']

        # Build the placeholders for the neighbourhood_depths for each feature type
        raw_array_placeholders = training.build_array_placeholders(FLAGS.training_batch_size, NEIGHBOURHOOD_SIZES, 1,
                                                                   all_feature_types)
        # Build the placeholder for the labels
        labels_placeholder = training.build_labels_placeholder(FLAGS.training_batch_size, FLAGS.classes_length)

        ################################################################################################################
        # Feeding
        ################################################################################################################

        feed_dict = {labels_placeholder: labels}

        for raw_array_placeholder, raw_array in zip(raw_array_placeholders, raw_arrays):
            for feature_type_name in list(feature_types.keys()):
                feed_dict[raw_array_placeholder[feature_type_name]] = raw_array[feature_type_name]

        ################################################################################################################
        # Tensorising
        ################################################################################################################

        # Any steps needed to get arrays ready for the rest of the pipeline
        with tf.name_scope('tensorising') as scope:
            # Tensorisors
            tensorisors = {'role_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                           'role_direction': lambda x: x,
                           'neighbour_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                           'neighbour_data_type': lambda x: x,
                           'neighbour_value_long': lambda x: x,
                           'neighbour_value_double': lambda x: x,
                           'neighbour_value_boolean': lambda x: x,
                           'neighbour_value_date': lambda x: x,
                           'neighbour_value_string': lambda x: x}

            tensorised_arrays = pp.preprocess_all(raw_array_placeholders, tensorisors)

        ################################################################################################################
        # Schema Traversals
        ################################################################################################################

        schema_traversal_executor = schema_ex.TraversalExecutor(self._tx)

        # THINGS
        thing_schema_traversal = trav.traverse_schema(self._traversal_strategies['thing'], schema_traversal_executor)

        # ROLES
        role_schema_traversal = trav.traverse_schema(self._traversal_strategies['role'], schema_traversal_executor)
        role_schema_traversal['has'] = ['has']

        ################################################################################################################
        # Encoders
        ################################################################################################################
        with tf.name_scope('encoding') as scope:
            thing_encoder = schema.MultiHotSchemaTypeEncoder(thing_schema_traversal)
            role_encoder = schema.MultiHotSchemaTypeEncoder(role_schema_traversal)

            # In case of issues https://github.com/tensorflow/hub/issues/61
            string_encoder = tf_hub.TensorFlowHubEncoder("https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1")

            data_types = list(data_ex.DATA_TYPE_NAMES)
            data_types.insert(0, NO_DATA_TYPE)  # For the case where an entity or relationship is encountered
            data_types_traversal = {data_type: data_types for data_type in data_types}

            # Later a hierarchy could be added to data_type meaning. e.g. long and double are both numeric
            data_type_encoder = schema.MultiHotSchemaTypeEncoder(data_types_traversal)

            encoders = {'role_type': role_encoder,
                        'role_direction': lambda x: x,
                        'neighbour_type': thing_encoder,
                        'neighbour_data_type': lambda x: data_type_encoder(tf.convert_to_tensor(x)),
                        'neighbour_value_long': lambda x: tf.to_float(x),
                        'neighbour_value_double': lambda x: x,
                        'neighbour_value_boolean': lambda x: tf.to_float(boolean.one_hot_boolean_encode(x)),
                        'neighbour_value_date': lambda x: tf.to_float(x),
                        'neighbour_value_string': string_encoder}

            encoded_arrays = encode.encode_all(tensorised_arrays, encoders)

        print('Encoded shapes')
        print([encoded_array.shape for encoded_array in encoded_arrays])

        training.supervised_train(neighbour_sample_sizes, encoded_arrays, labels)


if __name__ == "__main__":
    main()
