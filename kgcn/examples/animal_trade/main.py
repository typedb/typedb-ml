import time

import collections
import grakn
import numpy as np
import tensorflow as tf

import kgcn.examples.animal_trade.persistence as persistence
import kgcn.models.downstream as downstream
import kgcn.models.model as model
import kgcn.use_cases.attribute_prediction.label_extraction as label_extraction

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Enable debugging')
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

NUM_PER_CLASS = 2
BASE_PATH = f'data/{NUM_PER_CLASS}_concepts/'
flags.DEFINE_string('log_dir', BASE_PATH + 'out/out_' + TIMESTAMP, 'directory to use to store data from training')


def query_for_labelled_examples(tx, offset):
    appendix_vals = [1, 2, 3]

    concepts = []
    labels = []

    for a in appendix_vals:
        target_concept_query = f'match $e(exchanged-item: $t) isa exchange, has appendix $appendix; $appendix {a}; ' \
                               f'offset {offset}; limit {NUM_PER_CLASS}; get;'

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           ('t', collections.OrderedDict([('appendix', appendix_vals)]))
                                                           )
        concepts_with_labels = extractor(tx)
        if len(concepts_with_labels) == 0:
            raise RuntimeError(f'Couldn\'t find any concepts to match target query "{target_concept_query}"')

        concepts += [concepts_with_label[0] for concepts_with_label in concepts_with_labels]
        labels += [concepts_with_label[1]['appendix'] for concepts_with_label in concepts_with_labels]

    labels = np.array(labels, dtype=np.float32)
    return concepts, labels


def retrieve_persisted_labelled_examples(tx, file_path):

    concept_ids, labels = persistence.load_variable(file_path)
    print('==============================')
    print('Loaded concept IDs with labels')
    [print(concept_id, label) for concept_id, label in zip(concept_ids, labels)]
    concepts = []
    for concept_id in concept_ids:
        query = f'match $x id {concept_id}; get;'
        concept = next(tx.query(query)).get('x')
        concepts.append(concept)

    return concepts, labels


def main():
    keyspaces = {'train': "animaltrade_train",
                 'eval': "animaltrade_train",
                 'predict': "animaltrade_test"
                 }
    uri = "localhost:48555"
    sessions = {}
    txs = {}

    client = grakn.Grakn(uri=uri)

    labels_file_root = BASE_PATH + 'labels/labels_{}.p'

    find_and_save = [
        # 'train',
        # 'eval',
        # 'predict'
    ]
    run = True
    delete_all_labels_from_keyspace = False

    concepts_and_labels = {}

    save_input_data = {
        # 'train': tf.estimator.ModeKeys.TRAIN,
        # 'eval': tf.estimator.ModeKeys.EVAL,
        # 'predict': tf.estimator.ModeKeys.PREDICT
    }
    offsets = {
        'train': 0,
        'eval': NUM_PER_CLASS*2,
        'predict': 0
    }

    for keyspace_name in list(keyspaces.keys()):
        print(f'Concepts and labels for keyspace {keyspace_name}')
        sessions[keyspace_name] = client.session(keyspace=keyspaces[keyspace_name])
        txs[keyspace_name] = sessions[keyspace_name].transaction(grakn.TxType.WRITE)

        if keyspace_name in find_and_save:
            concepts_and_labels[keyspace_name] = query_for_labelled_examples(txs[keyspace_name], offsets[keyspace_name])
            concepts, labels = concepts_and_labels[keyspace_name]
            persistence.save_variable(([concept.id for concept in concepts], labels), labels_file_root.format(keyspace_name))

            # if delete_all_labels_from_keyspace:
            #     # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
            #     # that we are certain that they aren't being used by the learner
            #     print('Deleting concepts to avoid pollution')
            #     txs[keyspace_name].query(f'match $x isa appendix; delete $x;')
            #     txs[keyspace_name].commit()

        else:
            concepts_and_labels[keyspace_name] = retrieve_persisted_labelled_examples(txs[keyspace_name],
                                                                                      labels_file_root.format(keyspace_name))

    if (not find_and_save) and run:
        # neighbour_sample_sizes = (8, 2, 4)
        neighbour_sample_sizes = (2, 2)

        # storage_path=BASE_PATH+'input/'

        batch_size = buffer_size = NUM_PER_CLASS * FLAGS.num_classes
        kgcn = model.KGCN(neighbour_sample_sizes,
                          FLAGS.features_length,
                          FLAGS.starting_concepts_features_length,
                          FLAGS.aggregated_length,
                          FLAGS.output_length,
                          txs['train'],
                          batch_size,
                          buffer_size)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        classifier = downstream.SupervisedKGCNClassifier(kgcn, optimizer, FLAGS.num_classes, FLAGS.log_dir,
                                                         max_training_steps=FLAGS.max_training_steps)
        # Train
        train_concepts, train_labels = concepts_and_labels['train']
        classifier.train(sessions['train'], train_concepts, train_labels)

        # if len(save_input_data) > 0:
        #     for mode_name, mode in save_input_data.items():
        #         # Training data
        #         concepts, labels = concepts_and_labels[mode_name]
        #         kgcn.get_feed_dict(mode, sessions[mode_name], concepts, labels, save=True, load=False)
        # else:
        #     # Train
        #     # train_concepts, train_labels = concepts_and_labels['train']
        #     # kgcn.train(sessions['train'], train_concepts, train_labels)
        #     kgcn.train_from_file()
        #
        #     # Evaluate
        #     # eval_concepts, eval_labels = concepts_and_labels['eval']
        #     # kgcn.evaluate(sessions['eval'], eval_concepts, eval_labels)
        #     kgcn.evaluate_from_file()
        #
        #     # kgcn.predict(sessions['train'], predict_concepts)
        #     print('Evaluation on test set')
        #     kgcn.predict_from_file()

    for keyspace_name in list(keyspaces.keys()):
        # Close all transactions to clean up
        txs[keyspace_name].close()
        sessions[keyspace_name].close()


if __name__ == "__main__":
    main()
