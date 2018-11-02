import grakn
import tensorflow as tf

import kgcn.src.encoder.boolean as boolean
import kgcn.src.encoder.encode as encode
import kgcn.src.encoder.schema as schema
import kgcn.src.models.training as training
import kgcn.src.neighbourhood.data.concept as ci  # TODO Needs renaming from concept to avoid confusion
import kgcn.src.neighbourhood.data.executor as data_ex
import kgcn.src.neighbourhood.data.strategy as strat
import kgcn.src.neighbourhood.data.traversal as trv
import kgcn.src.neighbourhood.schema.executor as schema_ex
import kgcn.src.neighbourhood.schema.strategy as schema_strat
import kgcn.src.neighbourhood.schema.traversal as trav
import kgcn.src.preprocess.date_to_unixtime as date
import kgcn.src.preprocess.preprocess as pp
import kgcn.src.preprocess.raw_array_building as raw
import kgcn.src.neighbourhood.data.sampling.ordered as ordered


def main():
    tf.enable_eager_execution()
    # entity_query = "match $x isa person, has name 'Sundar Pichai'; get;"
    entity_query = "match $x isa company, has name 'Google'; get;"
    uri = "localhost:48555"
    keyspace = "test_schema"
    client = grakn.Grakn(uri=uri)
    session = client.session(keyspace=keyspace)
    tx = session.transaction(grakn.TxType.WRITE)

    neighbour_sample_sizes = (4, 3)
    sampler = ordered.ordered_sample

    # Strategies
    data_strategy = strat.DataTraversalStrategy(neighbour_sample_sizes, sampler)
    role_schema_strategy = schema_strat.SchemaRoleTraversalStrategy(include_implicit=True, include_metatypes=False)
    thing_schema_strategy = schema_strat.SchemaThingTraversalStrategy(include_implicit=True, include_metatypes=False)

    traversal_strategies = {'data': data_strategy,
                            'role': role_schema_strategy,
                            'thing': thing_schema_strategy}

    concepts = [concept.get('x') for concept in list(tx.query(entity_query))]

    kgcn = KGCN(tx, traversal_strategies)

    kgcn.model_fn(concepts, [1, 0])


class KGCN:

    def __init__(self, tx, traversal_strategies):
        self._tx = tx
        self._traversal_strategies = traversal_strategies

    def model_fn(self, concepts, labels=None):
        """
        A full Knowledge Graph Convolutional Network, running with TensorFlow and Grakn
        :return:
        """

        concept_infos = [ci.build_concept_info(concept) for concept in concepts]

        data_executor = data_ex.TraversalExecutor(self._tx)

        neighourhood_sampler = trv.NeighbourhoodTraverser(data_executor, self._traversal_strategies['data'])

        neighbourhood_depths = [neighourhood_sampler(concept_info) for concept_info in concept_infos]

        neighbour_roles = trv.concepts_with_neighbourhoods_to_neighbour_roles(neighbourhood_depths)

        ################################################################################################################
        # Raw Array Building
        ################################################################################################################

        raw_builder = raw.RawArrayBuilder(self._traversal_strategies['data'].neighbour_sample_sizes, len(concepts))
        raw_arrays = raw_builder.build_raw_arrays(neighbour_roles)

        ################################################################################################################
        # Preprocessing
        ################################################################################################################

        # Preprocessors
        preprocessors = {'role_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                         'role_direction': lambda x: x,
                         'neighbour_type': lambda x: tf.convert_to_tensor(x, dtype=tf.string),
                         'neighbour_data_type': lambda x: x,
                         'neighbour_value_long': lambda x: x,
                         'neighbour_value_double': lambda x: x,
                         'neighbour_value_boolean': lambda x: x,
                         'neighbour_value_date': date.datetime_to_unixtime,
                         'neighbour_value_string': lambda x: x}

        preprocessed_arrays = pp.preprocess_all(raw_arrays, preprocessors)

        ################################################################################################################
        # Schema Traversals
        ################################################################################################################

        schema_traversal_executor = schema_ex.TraversalExecutor(self._tx)
        # THINGS
        thing_schema_traversal = trav.traverse_schema(self._traversal_strategies['thing'], schema_traversal_executor)

        # ROLES
        role_schema_traversal = trav.traverse_schema(self._traversal_strategies['role'], schema_traversal_executor)

        ################################################################################################################
        # Encoders
        ################################################################################################################

        thing_encoder = schema.MultiHotSchemaTypeEncoder(thing_schema_traversal)
        role_encoder = schema.MultiHotSchemaTypeEncoder(role_schema_traversal)

        encoders = {'role_type': role_encoder,
                    'role_direction': lambda x: x,
                    'neighbour_type': thing_encoder,
                    'neighbour_data_type': lambda x: x,
                    'neighbour_value_long': lambda x: x,
                    'neighbour_value_double': lambda x: x,
                    'neighbour_value_boolean': lambda x: tf.cast(boolean.one_hot_boolean_encode(x), dtype=tf.float64),  # TODO Hacky, don't like it
                    'neighbour_value_date': lambda x: x,
                    'neighbour_value_string': lambda x: x}  # TODO Add actual string encoder

        encoded_arrays = encode.encode_all(preprocessed_arrays, encoders)

        training.supervised_train(encoded_arrays, labels)


if __name__ == "__main__":
    main()
