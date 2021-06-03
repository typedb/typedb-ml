[![Grabl](https://grabl.io/api/status/vaticle/kglib/badge.svg)](https://grabl.io/vaticle/kglib)
[![GitHub release](https://img.shields.io/github/release/vaticle/kglib.svg)](https://github.com/vaticle/typedb/releases/latest)
[![Discord](https://img.shields.io/discord/665254494820368395?color=7389D8&label=chat&logo=discord&logoColor=ffffff)](https://vaticle.com/discord)
[![Discussion Forum](https://img.shields.io/discourse/https/forum.vaticle.com/topics.svg)](https://forum.vaticle.com)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-typedb-796de3.svg)](https://stackoverflow.com/questions/tagged/typedb)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-typeql-3dce8c.svg)](https://stackoverflow.com/questions/tagged/typeql)

# TypeDB KGLIB (Knowledge Graph Library)

**KGLIB provides tools for machine learning with [TypeDB](https://github.com/vaticle/typedb). **

## Pipeline

The pipeline provided helps by allowing us to extract subgraphs from TypeDB. Each subgraph is a training example, which are sent to the learner in batches. Algorithms using this approach are scalable since they do not need to hold the whole graph in memory for training.

The pipeline is as follows:
1. Extract data from `TypeDB` into Python [NetworkX](https://networkx.org) in-memory subgraphs by specifying multiple [TypeQL](https://github.com/vaticle/typeql) queries.
2. Encode the nodes and edges of the NetworkX graphs
3. Either (a) transform the encoded values into features, ready for input into a graph/geometric learning pipeline (for example the upcoming PyTorch implementation); or (b) Embed the encoded values according to the Types present in your database (TensorFlow only, PyTorch coming soon). This type-centric embedding is crucial to extracting the context explicitly captured in TypeDB's Type System. 
4. Feed the features to a learning algorithm (see below)
5. Optionally, store the predictions made by the learner in TypeDB. These predictions can then be queried using TypeQL. This means we can trivially run more learning tasks over the knowledge base, including the newly made predictions. This is knowledge graph completion.

## Learning Algorithms 
This repo contains one algorithmic implementation: [*Knowledge Graph Convolutional Network* (KGCN)](kglib/kgcn_tensorflow). This is a generic method for relation predication over any TypeDB database. There is a [full worked example](kglib/kgcn_tensorflow/examples/diagnosis) and an explanation of how the approach works.

You are encouraged to use the tools available in KGLIB to interface TypeDB to your own algorithmic implementations, or to use/leverage prebuilt implementations available from popular libraries such as [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) or [Graph Nets](https://github.com/deepmind/graph_nets) (TensorFlow/Sonnet). 

## Quickstart
**Requirements**

- Python >= 3.6, <= 3.7.x (TensorFlow 1.14.0 doesn't support later Python versions) .

- KGLIB installed via pip: `pip install typedb-kglib`. 

- [TypeDB 2.1.1](https://github.com/vaticle/typedb/releases) running in the background.

- `typedb-client-python` 2.1.0 ([PyPi](https://pypi.org/project/typedb-client/), [GitHub release](https://github.com/vaticle/typedb-client-python/releases)). This should be installed automatically when you `pip install typedb-kglib`.

**Run**
Take a look at [*Knowledge Graph Convolutional Networks* (KGCNs)](kglib/kgcn_tensorflow) to see a walkthrough of how to use the library.

**Building from source**

It's expected that you will use `pip` to install, but should you need to make your own changes to the library, and import it into your project, you can build from source as follows.

Clone KGLIB:

```
git clone git@github.com:vaticle/kglib.git
```

`cd` in to the project:

```
cd kglib
```

Build all targets:

```
bazel build //...
```

Run all tests. Requires Python 3.6+ on your `PATH`. Test dependencies are for Linux since that is the CI environment: 

```
bazel test //kglib/... --test_output=streamed --spawn_strategy=standalone --action_env=PATH
```

Build the pip distribution. Outputs to `bazel-bin`:

```
bazel build //:assemble-pip
```

## Development

To follow the development conversation, please join the [Vaticle Discord](https://discord.com/invite/grakn), and join the `#kglib` channel. Alternatively, start a new topic on the [Vaticle Discussion Forum](https://forum.vaticle.com).

KGLIB requires that you have migrated your data into a [TypeDB](https://github.com/vaticle/typedb) or TypeDB Cluster instance. There is an [official examples repo](https://github.com/vaticle/examples) for how to go about this, and information available on [migration in the docs](https://docs.vaticle.com/docs/examples/phone-calls-migration-python). Alternatively, there are fantastic community-led projects growing in the [TypeDB OSI](https://typedb.org) to facilitate fast and easy data loading.
