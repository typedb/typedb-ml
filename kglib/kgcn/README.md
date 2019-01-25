# Knowledge Graph Convolutional Networks (KGCNs)

This project introduces a novel model: the *Knowledge Graph Convolutional Network* (KGCN). The principal idea of this work is to forge a bridge between knowledge graphs and machine learning, using [Grakn](https://github.com/graknlabs/grakn) as the knowledge graph. A KGCN can be used to create vector representations, *embeddings*, of any labelled set of Grakn Concepts via supervised learning. As a result, a KGCN can be trained directly for the classification or regression of Concepts stored in Grakn. Future work will include building embeddings via unsupervised learning.![KGCN Process](readme_images/KGCN_process.png)



## Methodology

The ideology behind this project is described [here](https://blog.grakn.ai/knowledge-graph-convolutional-networks-machine-learning-over-reasoned-knowledge-9eb5ce5e0f68). The principles of the implementation are based on [GraphSAGE](http://snap.stanford.edu/graphsage/), from the Stanford SNAP group, made to work over a **knowledge graph**. Instead of working on a typical property graph, a KGCN learns from the context of a *typed hypergraph*, **Grakn**. Additionally, it learns from facts deduced by Grakn's *automated logical reasoner*. From this point onwards some understanding of [Grakn's docs](http://dev.grakn.ai) is assumed.

#### How do KGCNs work?

The purpose of this method is to derive embeddings for a set of Concepts (and thereby directly learn to classify them). We start by querying Grakn to find a set of labelled examples. Following that, we gather data about the neighbourhood of each example Concept. We do this by considering their *k-hop* neighbours.

![k-hop neighbours](readme_images/k-hop_neighbours.png)We retrieve the data concerning this neighbourhood from Grakn. This information includes the *type hierarchy*, *roles*, and *attribute* values of each neighbouring Concept encountered.

To create embeddings, we build a network in TensorFlow that successively aggregates and combines features from the K hops until a 'summary' representation remains - an embedding. In our example these embeddings are directly optimised to perform multi-class classification. This is achieved by passing the embeddings to a single subsequent dense layer and determining loss via softmax cross entropy with the labels retrieved.

![Aggregation and Combination process](readme_images/aggregate_and_combine.png)



## Usage by example - CITES Animal Trade Data

### Quickstart

**Requirements:**

- Python 3.6.3 or higher
- kglib installed from pip: `pip install --extra-index-url https://test.pypi.org/simple/ grakn-kglib`
- The `grakn-animaltrade.zip` dataset from the [latest release](https://github.com/graknlabs/kglib/releases/latest). This is a dataset that has been pre-loaded into Grakn v1.5 (so you don't have to run the data import yourself), with two keyspaces: `animaltrade_train` and `animaltrade_test`.

**To use:**

- Prepare the data:

  - If you already have an insatnce of Grakn running, make sure to stop it using `./grakn server stop`
  
  - Download the pre-loaded Grakn distribution from the [latest release](https://github.com/graknlabs/kglib/releases/latest)

  - Unzip the distribution `unzip grakn-animaltrade.zip `, where you store this doesn't matter 

  - cd into the distribution `cd grakn-animaltrade`
  
  - start Grakn `./grakn server start`

  - Confirm that the training keyspace is present and contains data 

    `./grakn console -k animaltrade_train`

    `match $t isa traded-item; limit 1; get;`

    and then `exit`

- Run the `main` function of the example: 

  `cd kglib`

  `python3 -m kglib.kgcn.examples.animal_trade.main`

  This will run the full pipeline: retrieving data, building and training a KGCN classifier

#### Details

The CITES dataset details exchanges of animal-based products between countries. In this example we aim to predict the value of `appendix` for a set of samples. This `appendix` can be thought of as the level of endangerment that a `traded-item` is subject to, where `1` represents the highest level of endangerment, and `3` the lowest.

The [main](examples/animal_trade/main.py) function will:

- Search Grakn for 30 concepts (with attributes as labels) to use as the training set, 30 for the evaluation set, and 30 for the prediction set using queries such as (limiting the returned stream):

  ```
  match $e(exchanged-item: $traded-item) isa exchange, has appendix $appendix; $appendix 1; get;
  ```

  This searches for an `exchange` between countries that has an `appendix` (endangerment level) of `1`, and finds the `traded-item` that was exchanged

- Save those labelled samples to file

- Delete all `appendix` attributes from both `animaltrade_train` and `animaltrade_test` keyspaces. This is the label we will predict in this example, so it should not be present in Grakn otherwise the network can cheat

- Search Grakn for the k-hop neighbours of the selected examples, and store information about them as arrays, demoted in the code as `raw_arrays`. This data is saved to file so that subsequent steps can be re-run without recomputing these data

- Build the TensorFlow computation graph using `model.KGCN`, including a multi-class classification step and learning procedure defined by `downstream.SupervisedKGCNClassifier`

- Feed the `raw_arrays` to the TensorFlow graph, and performs learning