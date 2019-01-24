# Knowledge Graph Convolutional Networks (KGCNs)

This project introduces a novel model: the Knowledge Graph Convolutional Network. The principal idea of this work is to build a bridge between knowledge graphs and machine learning. KGCNs can be used to create vector representations or *embeddings* of any labelled set of Grakn concepts. As a result, a KGCN can be trained directly for the classification or regression of Concepts stored in Grakn. Future work will include building embeddings via unsupervised learning.

![Screenshot 2018-10-23 at 15.12.46](/Users/jamesfletcher/Dropbox (Grakn)/James/Research/KGCN Introductory Blog Post/imagery/Screenshot 2018-10-23 at 15.12.46.png)

## Methodology

The ideology behind this project is described [here](https://blog.grakn.ai/knowledge-graph-convolutional-networks-machine-learning-over-reasoned-knowledge-9eb5ce5e0f68). The principles of the implementation are based on [GraphSAGE](http://snap.stanford.edu/graphsage/), from the Stanford SNAP group, made to work over a **knowledge graph**. Instead of working on a typical property graph, a KGCN learns from the context of a *typed hypergraph*, Grakn. Additionally, it learns from facts deduced by Grakn's *automated logical reasoner*. From this point on some understanding of [Grakn's docs](http://dev.grakn.ai) is assumed.

####How does a KGCN work?

The purpose of this method is to derive embeddings for a set of Concepts (and thereby directly learn to classify them). We start by querying Grakn to find a set of examples with labels. Following that, we gather data about the neighbourhood of each example Concept. We do this by considering their *k-hop* neighbours.

![Screenshot 2019-01-24 at 19.00.31](/Users/jamesfletcher/Desktop/screenshots/Screenshot 2019-01-24 at 19.00.31.png)We retrieve the data concerning this neighbourhood from Grakn. This includes information on the *types*, *roles*, and *attribute* values of each neighbour encountered.

To create embeddings, we build a network in TensorFlow that successively aggregates and combines features from the K hops until a 'summary' representation remains - an embedding. In our example these embeddings are directly optimised to perform multi-class classification via a single subsequent dense layer and softmax cross entropy.

![Screenshot 2019-01-24 at 19.03.08](/Users/jamesfletcher/Desktop/screenshots/Screenshot 2019-01-24 at 19.03.08.png)



##Example - CITES Animal Trade Data

####Quickstart

**Requirements:**

- Python 3.6.3 or higher

- kglib installed from pip: `pip install --extra-index-url https://test.pypi.org/simple/ grakn-kglib`
- The `animaltrade` dataset from the latest release. This is a dataset that has been pre-loaded into Grakn v1.5 (so you don't have to run the data import yourself), with two keyspaces: `animaltrade_train` and `animaltrade_test`.

**To use:**

- Prepare the data:

  - If you already have an insatnce of Grakn running, make sure to stop it using `./grakn server stop`

  - Unzip the pre-loaded Grakn + dataset from the latest release, the location you store this in doesn't matter

  - `cd` into the dataset and start Grakn: `./grakn server start`

  - Confirm that the training keyspace is present and contains data 

    `./grakn console -k animaltrade_train`

    `match $t isa traded-item; limit 1; get;`

    and then `exit`

- Run the `main` function of the example: 

  `cd kglib`

  `python3 -m kglib.kgcn.examples.animal_trade.main`