[![GitHub release](https://img.shields.io/github/release/vaticle/typedb-ml.svg)](https://github.com/vaticle/typedb/releases/latest)
[![Discord](https://img.shields.io/discord/665254494820368395?color=7389D8&label=chat&logo=discord&logoColor=ffffff)](https://vaticle.com/discord)
[![Discussion Forum](https://img.shields.io/discourse/https/forum.vaticle.com/topics.svg)](https://forum.vaticle.com)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-typedb-796de3.svg)](https://stackoverflow.com/questions/tagged/typedb)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-typeql-3dce8c.svg)](https://stackoverflow.com/questions/tagged/typeql)

# TypeDB-ML
_Previously known as KGLIB._

**TypeDB-ML provides tools to enable graph algorithms and machine learning with [TypeDB](https://github.com/vaticle/typedb).**

There are integrations for [NetworkX](https://networkx.org) and for [PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric).

[NetworkX](https://networkx.org) integration allows you to use a [large library of algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html) over graph data exported from TypeDB.

[PyTorch Geometric (PyG)](https://github.com/pyg-team/pytorch_geometric) integration gives you a toolbox to build Graph Neural Networks (GNNs) for your TypeDB data, with an example included for link prediction (or: binary relation prediction, in TypeDB terms). The structure of the GNNs are totally customisable, with network components for popular topics such as graph attention and graph transformers built-in.  

## Features

### NetworkX
- Declare the graph structure of your queries, with optional sampling functions.
- Query a TypeDB instance and combine many results across many queries into a single graph (`build_graph_from_queries`).
### PyTorch Geometric
- A `DataSet` object to lazily load graphs from a TypeDB instance. Each graph is converted to a PyG `Data` object.
- It's most natural to work with PyG `HeteroData` objects since all data in TypeDB has a type. Conversion from `Data` to `HeteroData`is available in PyG, but it loses node ordering information. To remedy this, TypeDB-ML provides `store_concepts_by_type` to store concepts consistent with a `HeteroData` object. This enables concepts to be properly re-associated with predictions after learning is finished.
- A `FeatureEncoder` to orchestrate encoders to generate features for graphs.
- Encoders for Continuous and Categorical values to apply encodings/embedding spaces to the types and attribute values present in TypeDB data.
- A [full example for link prediction](examples/diagnosis)
### Other
- Example usage of Tensorboard for PyG `HeteroData`

## Resources
You may find the following resources useful, particularly to understand why TypeDB-ML started: 
- [Strongly Typed Data for Machine Learning](https://www.youtube.com/watch?v=qhUyurWMiSQ) (YouTube, 2021)
- [How Can We Complete a Knowledge Graph?](https://www.youtube.com/watch?v=nYDi1_UaFtU) (YouTube, 2018)

## Quickstart

### Install

- Python >= 3.7.x

- Grab the `requirements.txt` file from [here](requirements.txt) and install the requirements with `pip install -r requirements.txt`. This is due to some intricacies installing PyG's dependencies, see [here](https://github.com/pyg-team/pytorch_geometric/issues/861) for details.

- Installed TypeDB-ML: `pip install typedb-ml`. 

- [TypeDB 2.11.1](https://github.com/vaticle/typedb/releases) running in the background.

- `typedb-client-python` 2.11.x ([PyPi](https://pypi.org/project/typedb-client/), [GitHub release](https://github.com/vaticle/typedb-client-python/releases)). This should be installed automatically when you `pip install typedb-ml`.

### Run the Example

Take a look at the [PyTorch Geometric heterogeneous link prediction example](examples/diagnosis) to see how to use TypeDB-ML to build a GNN on TypeDB data.

## Development

To follow the development conversation, please join the [Vaticle Discord](https://discord.com/invite/vaticle), and join the `#typedb-ml` channel. Alternatively, start a new topic on the [Vaticle Discussion Forum](https://forum.vaticle.com).

TypeDB-ML requires that you have migrated your data into a [TypeDB](https://github.com/vaticle/typedb) or TypeDB 
Cluster instance. There is an [official examples repo](https://github.com/vaticle/examples) for how to go about this, and information available on [migration in the docs](https://docs.vaticle.com/docs/examples/phone-calls-migration-python). Alternatively, there are fantastic community-led projects growing in the [TypeDB OSI](https://typedb.org) to facilitate fast and easy data loading, for example [TypeDB Loader](https://github.com/typedb-osi/typedb-loader).


### Building from Source

It's expected that you will use Pip to install, but should you need to make your own changes to the library, and import it into your project, you can build from source as follows:

Clone TypeDB-ML:

```
git clone git@github.com:vaticle/typedb-ml.git
```

Go into the project directory:

```
cd typedb-ml
```

Build all targets:

```
bazel build //...
```

Run all tests. Requires Python 3.7+ on your `PATH`. Test dependencies are for Linux since that is the CI environment: 

```
bazel test //typedb_ml/... --test_output=streamed --spawn_strategy=standalone --action_env=PATH
```

Build the pip distribution. Outputs to `bazel-bin`:

```
bazel build //:assemble-pip
```
