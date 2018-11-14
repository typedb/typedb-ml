import collections

import grakn
import numpy as np

import kgcn.src.models.model as model
import kgcn.src.neighbourhood.data.sampling.ordered as ordered
import kgcn.src.neighbourhood.data.sampling.sampler as samp
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.use_cases.attribute_prediction.label_extraction as label_extraction


def main():
    keyspaces = {'train': "animaltrade_train",
                 'eval': "animaltrade_test",
                 'predict': "animaltrade_test"}
    uri = "localhost:48555"
    txs = {}

    client = grakn.Grakn(uri=uri)
    train_session = client.session(keyspace=keyspaces['train'])
    txs['train'] = train_session.transaction(grakn.TxType.WRITE)

    appendix_vals = [1, 2, 3]

    concepts = []
    labels = []

    for a in appendix_vals:
        target_concept_query = f"match $x isa exchange, has appendix $appendix; $appendix {a}; limit 10; get;"

        extractor = label_extraction.ConceptLabelExtractor(target_concept_query,
                                                           ('x', collections.OrderedDict([('appendix', appendix_vals)]))
                                                           )
        concepts_with_labels = extractor(txs['train'])

        concepts += [concepts_with_label[0] for concepts_with_label in concepts_with_labels]
        labels += [concepts_with_label[1]['appendix'] for concepts_with_label in concepts_with_labels]

    labels = np.array(labels, dtype=np.float32)

    neighbour_sample_sizes = (2, 1)

    sampling_method = ordered.ordered_sample

    samplers = []
    for sample_size in neighbour_sample_sizes:
        samplers.append(samp.Sampler(sample_size, sampling_method, limit=sample_size + 1))

    # Strategies
    role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=False, include_metatypes=False)
    thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=False, include_metatypes=False)

    traversal_strategies = {'role': role_schema_strategy,
                            'thing': thing_schema_strategy}

    kgcn = model.KGCN(txs['train'], traversal_strategies, samplers)

    kgcn.train(txs['train'], concepts, labels)
    kgcn.predict(txs['train'], concepts)


if __name__ == "__main__":
    main()