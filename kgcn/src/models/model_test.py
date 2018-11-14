import unittest


class TestKGCN(unittest.TestCase):
    """
    This is one possible implementation of a Knowledge Graph Convolutional Network (KGCN). This implementation uses
    ideas of sampling and aggregating the information of the connected neighbours of a concept in order to build a
    representation of that concept.

    As a user we want to provide parameters to the model, including:
    - Number of neighbours to sample at each depth
    - How to sample those neighbours (incl. pseudo-random params)
    - Whether to propagate sampling through attributes
    - Whether to use implicit relationships, or 'has' roles
    - Number of training steps
    - Learning rate
    - Optimiser e.g. AdamOptimiser

    Then we want to provide concepts to train, evaluate and perform prediction upon. In the case of supervised
    learning we also need to provide labels for those concepts for training and evaluation.

    We want to be able to re-run the model without re-running the Grakn traversals. This requires some way to persist
    the data acquired from the Grakn traversals and interrupt the pipeline and continue later. Probably best via
    TensorFlow checkpoints.

    Each stage of training should be bound to a keyspace, and these keyspaces may differ. For example, we will want
    to use a separate keyspace for training to that used for evaluation, or we may have crossover between our
    training and evaluation data
    """

    def setUp(self):
        self._kgcn = KGCN(model_params, traversal_strategies, traversal_samplers)

    def test_train(self):
        self._kgcn.train(tx, concepts, labels)

    def test_evaluate(self):
        self._kgcn.evaluate(tx, concepts, labels)

    def test_predict(self):
        predicted_labels = self._kgcn.predict(tx, concepts)
