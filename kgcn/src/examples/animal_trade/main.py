import time

import collections
import grakn
import numpy as np
import tensorflow as tf

import kgcn.src.examples.animal_trade.persistence as persistence
import kgcn.src.models.model as model
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.use_cases.attribute_prediction.label_extraction as label_extraction

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_boolean('debug', False, 'Enable debugging')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
flags.DEFINE_integer('classes_length', 3, 'Number of classes')
flags.DEFINE_integer('features_length', 192, 'Number of features after encoding')
flags.DEFINE_integer('starting_concepts_features_length', 173,
                     'Number of features after encoding for the nodes of interest, which excludes the features for '
                     'role_type and role_direction')
flags.DEFINE_integer('aggregated_length', 20, 'Length of aggregated representation of neighbours, a hidden dimension')
flags.DEFINE_integer('output_length', 32, 'Length of the output of "combine" operation, taking place at each depth, '
                                          'and the final length of the embeddings')

flags.DEFINE_integer('max_training_steps', 10000, 'Max number of gradient steps to take during gradient descent')

TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")

NUM_PER_CLASS = 45
BASE_PATH = f'data/{NUM_PER_CLASS}_concepts/'
flags.DEFINE_string('log_dir', BASE_PATH + 'out/out_' + TIMESTAMP, 'directory to use to store data from training')


def query_for_labelled_examples(tx):
    appendix_vals = [1, 2, 3]

    concepts = []
    labels = []

    for a in appendix_vals:
        target_concept_query = f'match $x isa exchange, has appendix $appendix; $appendix {a}; limit ' \
                               f'{NUM_PER_CLASS}; get;'

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           ('x', collections.OrderedDict([('appendix', appendix_vals)]))
                                                           )
        concepts_with_labels = extractor(tx)

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
                 'eval': "animaltrade_test",
                 # 'predict': "animaltrade_test"
                 }
    uri = "localhost:48555"
    sessions = {}
    txs = {}

    client = grakn.Grakn(uri=uri)

    labels_file_root = BASE_PATH + 'labels/labels_{}.p'

    find_and_save = [
        # 'train',
        # 'eval'
    ]
    run = True
    delete_all_labels_from_keyspace = True

    concepts_and_labels = {}

    save_data = {
        # 'train': tf.estimator.ModeKeys.TRAIN,
        # 'eval': tf.estimator.ModeKeys.EVAL
    }

    for keyspace_name in list(keyspaces.keys()):
        print(f'Concepts and labels for keyspace {keyspace_name}')
        sessions[keyspace_name] = client.session(keyspace=keyspaces[keyspace_name])
        txs[keyspace_name] = sessions[keyspace_name].transaction(grakn.TxType.WRITE)

        if keyspace_name in find_and_save:
            concepts_and_labels[keyspace_name] = query_for_labelled_examples(txs[keyspace_name])
            concepts, labels = concepts_and_labels[keyspace_name]
            persistence.save_variable(([concept.id for concept in concepts], labels), labels_file_root.format(keyspace_name))

            if delete_all_labels_from_keyspace:
                # Once concept ids have been stored with labels, then the labels stored in Grakn can be deleted so
                # that we are certain that they aren't being used by the learner
                print('Deleting concepts to avoid pollution')
                txs[keyspace_name].query(f'match $x isa appendix; delete $x;')
                txs[keyspace_name].commit()

        else:
            concepts_and_labels[keyspace_name] = retrieve_persisted_labelled_examples(txs[keyspace_name],
                                                                                      labels_file_root.format(keyspace_name))

    if (not find_and_save) and run:
        neighbour_sample_sizes = (5, 5)

        sampling_method = ordered.ordered_sample

        samplers = []
        for sample_size in neighbour_sample_sizes:
            samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size + 1))

        # Strategies
        role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=False, include_metatypes=False)
        thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=False, include_metatypes=False)

        traversal_strategies = {'role': role_schema_strategy,
                                'thing': thing_schema_strategy}

        kgcn = model.KGCN(txs['train'], traversal_strategies, samplers, storage_path=BASE_PATH+'input/')

        if len(save_data) > 0:
            for mode_name, mode in save_data.items():
                # Training data
                concepts, labels = concepts_and_labels[mode_name]
                kgcn.get_feed_dict(mode, txs[mode_name], concepts, labels, save=True, load=False)
        else:
            # Train
            # train_concepts, train_labels = concepts_and_labels['train']
            # kgcn.train(txs['train'], train_concepts, train_labels)
            kgcn.train_from_file()

            # Evaluate
            # eval_concepts, eval_labels = concepts_and_labels['eval']
            # kgcn.evaluate(txs['eval'], eval_concepts, eval_labels)
            kgcn.evaluate_from_file()

            # kgcn.predict(txs['train'], predict_concepts)


if __name__ == "__main__":
    main()
