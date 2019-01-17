import sys
import time

import grakn
import tensorflow as tf

import kgcn.models.downstream as downstream
import kgcn.models.model as model
import kgcn.preprocess.persistence as persistence
import kgcn.utils as utils
from kgcn.utils import retrieve_persisted_labelled_concepts, query_for_random_examples_with_attribute

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('num_classes', 3, 'Number of classes')
flags.DEFINE_integer('features_length', 198, 'Number of features after encoding')
flags.DEFINE_integer('starting_concepts_features_length', 173,
                     'Number of features after encoding for the nodes of interest, which excludes the features for '
                     'role_type and role_direction')
flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
                                          'and the final length of the embeddings')

flags.DEFINE_integer('max_training_steps', 10000, 'Max number of gradient steps to take during gradient descent')

TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")

NUM_PER_CLASS = 30
POPULATION_SIZE_PER_CLASS = 1000
# BASE_PATH = f'dataset/{NUM_PER_CLASS}_concepts/'
BASE_PATH = 'dataset/30_concepts_ordered_7x2x2_without_attributes_with_rules/'
flags.DEFINE_string('log_dir', BASE_PATH + 'out/out_' + TIMESTAMP, 'directory to use to store data from training')

TRAIN = 'train'
EVAL = 'eval'
PREDICT = 'predict'

KEYSPACES = {
    TRAIN: "animaltrade_train",
    EVAL: "animaltrade_train",
    PREDICT: "animaltrade_test"
}

URI = "localhost:48555"

NEIGHBOUR_SAMPLE_SIZES = (7, 2, 2)  # TODO (7, 2, 2) Throws an error without rules

sys.stdout = utils.Logger(FLAGS.log_dir + '/output.log')


def combine_labelled_examples(concepts_dict, labels_dict):
    concepts = []
    labels = []
    for key in concepts_dict.keys():
        concepts.extend(concepts_dict[key])
        labels.extend(labels_dict[key])
    return concepts, labels


def find_and_save_labelled_concepts(examples_query, transactions, saved_labels_path):
    concepts = {}
    labels = {}

    print(f'Finding concepts and labels')
    print('    for training and evaluation')
    concepts_dicts, labels_dicts = query_for_random_examples_with_attribute(transactions[TRAIN], examples_query,
                                                                            'traded-item', 'appendix', [1, 2, 3],
                                                                            NUM_PER_CLASS * 2,
                                                                            POPULATION_SIZE_PER_CLASS * 2)

    half = NUM_PER_CLASS
    # Iterate over the classes
    concepts[TRAIN] = []
    concepts[EVAL] = []
    labels[TRAIN] = []
    labels[EVAL] = []
    for key in concepts_dicts.keys():
        concepts[TRAIN].extend(concepts_dicts[key][:half])
        concepts[EVAL].extend(concepts_dicts[key][half:])
        labels[TRAIN].extend(labels_dicts[key][:half])
        labels[EVAL].extend(labels_dicts[key][half:])

    print('    for prediction')
    concepts_dicts_predict, labels_dicts_predict = query_for_random_examples_with_attribute(transactions[TRAIN],
                                                                                            examples_query,
                                                                                            'traded-item',
                                                                                            'appendix', [1, 2, 3],
                                                                                            NUM_PER_CLASS,
                                                                                            POPULATION_SIZE_PER_CLASS)
    concepts[PREDICT], labels[PREDICT] = combine_labelled_examples(concepts_dicts_predict, labels_dicts_predict)

    for keyspace_key in list(KEYSPACES.keys()):
        persistence.save_variable(([concept.id for concept in concepts[keyspace_key]], labels[keyspace_key]),
                                  saved_labels_path.format(keyspace_key))
    return concepts, labels


def delete_all_labels_from_keyspace(tx):
    # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
    # that we are certain that they aren't being used by the learner
    print('Deleting concepts to avoid data pollution')
    tx.query(f'match $x isa appendix; delete $x;')
    tx.commit()


def load_saved_labelled_concepts(transactions, saved_labels_path):
    concepts = {}
    labels = {}
    for keyspace_key in list(KEYSPACES.keys()):
        print(f'Loading concepts and labels for {keyspace_key}')
        concepts[keyspace_key], labels[keyspace_key] = retrieve_persisted_labelled_concepts(transactions[keyspace_key],
                                                                                            saved_labels_path.format(
                                                                                                keyspace_key))
    return concepts, labels


def get_sessions():
    for keyspace_key in list(KEYSPACES.keys()):
        sessions[keyspace_key] = client.session(keyspace=KEYSPACES[keyspace_key])
        transactions[keyspace_key] = sessions[keyspace_key].transaction(grakn.TxType.WRITE)


def main():

    sessions = {}
    transactions = {}

    examples_query = 'match $e(exchanged-item: $traded-item) isa exchange, has appendix $appendix; $appendix {}; ' \
                     'limit {}; get;'

    client = grakn.Grakn(uri=URI)

    saved_labels_path = BASE_PATH + 'labels/labels_{}.p'

    find_labelled_concepts = False
    delete_all_labels_from_keyspace = False
    run = True
    save_input_data = False  # Overwrites any saved data

    # Check if saved concepts and labels exist, and act accordingly
    if concepts_saved:
        # In this case any params to describe how to pick examples become invalid
        concepts, labels = load_saved_labelled_concepts()
    else:
        concepts, labels = find_and_save_labelled_concepts()
        delete_all_labels_from_keyspace()

    feed_dict_storer = persistence.FeedDictStorer(BASE_PATH+'input/')

    if (not find_labelled_concepts) and run:

        # storage_path=BASE_PATH+'input/'

        batch_size = buffer_size = NUM_PER_CLASS * FLAGS.num_classes
        kgcn = model.KGCN(NEIGHBOUR_SAMPLE_SIZES,
                          FLAGS.features_length,
                          FLAGS.starting_concepts_features_length,
                          FLAGS.aggregated_length,
                          FLAGS.output_length,
                          transactions[TRAIN],
                          batch_size,
                          buffer_size,
                          # sampling_method=random_sampling.random_sample,
                          # sampling_limit_factor=4
                          )

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        classifier = downstream.SupervisedKGCNClassifier(kgcn, optimizer, FLAGS.num_classes, FLAGS.log_dir,
                                                         max_training_steps=FLAGS.max_training_steps)

        feed_dict = {}

        if save_input_data:

            feed_dict[TRAIN] = classifier.get_feed_dict(sessions[TRAIN], concepts[TRAIN], labels=labels[TRAIN])
            feed_dict_storer.store_feed_dict(TRAIN, feed_dict[TRAIN])

            feed_dict[EVAL] = classifier.get_feed_dict(sessions[EVAL], concepts[EVAL], labels=labels[EVAL])
            feed_dict_storer.store_feed_dict(EVAL, feed_dict[EVAL])

        else:
            feed_dict[TRAIN] = feed_dict_storer.retrieve_feed_dict(TRAIN)
            feed_dict[EVAL] = feed_dict_storer.retrieve_feed_dict(EVAL)

        # Train
        classifier.train(feed_dict[TRAIN])

        # Eval
        classifier.eval(feed_dict[EVAL])

    for keyspace_key in list(KEYSPACES.keys()):
        # Close all transactions to clean up
        transactions[keyspace_key].close()
        sessions[keyspace_key].close()


if __name__ == "__main__":
    main()
