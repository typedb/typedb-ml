# Grakn KGLIB (Knowledge Graph Library)

[![CircleCI](https://circleci.com/gh/graknlabs/kglib/tree/master.svg?style=shield)](https://circleci.com/gh/graknlabs/kglib/tree/master)
[![Slack Status](http://grakn-slackin.herokuapp.com/badge.svg)](https://grakn.ai/slack)
[![Discussion Forum](https://img.shields.io/discourse/https/discuss.grakn.ai/topics.svg)](https://discuss.grakn.ai)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-grakn-796de3.svg)](https://stackoverflow.com/questions/tagged/grakn)
[![Stack Overflow](https://img.shields.io/badge/stackoverflow-graql-3dce8c.svg)](https://stackoverflow.com/questions/tagged/graql)

[Grakn](https://github.com/graknlabs/grakn) lets us create Knowledge Graphs from our data. But what challenges do we encounter where querying alone won't cut it? What library can address these challenges?

To respond to these scenarios, KGLIB is the centre of all research projects conducted at Grakn Labs. In particular, its focus is on the integration of machine learning with the Grakn Knowledge Graph. More on this below, in [*Knowledge Graph Tasks*](https://github.com/graknlabs/kglib#knowledge-graph-tasks).

At present this repo contains one project: [*Knowledge Graph Convolutional Networks* (KGCNs)](https://github.com/graknlabs/kglib/tree/master/kglib/kgcn). Go there for more info on getting started with a working example.

## Quickstart
**Requirements**

- Python >= 3.6

- KGLIB installed via pip: `pip install grakn-kglib`. 

- [Grakn Core 1.8.0](https://github.com/graknlabs/grakn/releases) running in the background

- client-python 1.8.0 ([PyPi](https://pypi.org/project/grakn-client/), [GitHub release](https://github.com/graknlabs/client-python/releases))

**Run**
Take a look at [*Knowledge Graph Convolutional Networks* (KGCNs)](https://github.com/graknlabs/kglib/tree/master/kglib/kgcn) to see a walkthrough of how to use the library.

**Building from source**

Clone KGLIB:

```
git clone git@github.com:graknlabs/kglib.git
```

`cd` in to the project:

```
cd kglib
```

To build all targets can be built:

```
bazel build //...
```

To run all tests (requires Python 3.6+): 

```
bazel test //kglib/... --test_output=streamed --spawn_strategy=standalone --python_version PY3 --python_path $(which python3)
```

To build the pip distribution (find the output in `bazel-bin`):

```
bazel build //:assemble-pip
```

## Knowledge Graph Tasks

Below is a set of tasks to be conducted over Knowledge Graphs (KGs) that we have identified from real Grakn use cases. The objective of KGLIB is to implement a portfolio of solutions for these tasks for Grakn Knowledge Graphs.

- Relation Prediction (a.k.a. Link Prediction)
- Attribute Prediction
- Subgraph Prediction
- Building Concept Embeddings
- Rule Mining (a.k.a. Association Rule Learning)
- Ontology Merging
- Automated Knowledge Graph Creation
- Expert Systems
- Optimal Pattern Finding
- System Design Automation and Configuration Automation
- Fuzzy Pattern Matching
- Querying and Responding via Natural Language

*Many of these tasks are open research problems, thus far "unsolved" for the general case.*

We describe these tasks in more detail below. Where a solution is readily available in KGLIB, it is listed against the relevant task(s).

We openly invite collaboration to solve these problems! All contributions are welcome - code, issues, ideas, discussions, pointers to existing tools, and relevant datasets will all help this project evolve!

If you wish to discuss your ideas more conversationally, and to follow the development conversation, please join the [Grakn Slack](https://grakn.ai/slack), and join the #kglib channel. Alternatively, start a new topic on the [Grakn Discussion Forum](https://discuss.grakn.ai).

All of the solutions in KGLIB require that you have migrated your data into a [Grakn Core](https://github.com/graknlabs/grakn) or [Grakn KGMS](https://grakn.ai/grakn-kgms) instance. There is an [official examples repo](https://github.com/graknlabs/examples) for how to go about this, and information available on [migration in the Grakn docs](https://dev.grakn.ai/docs/examples/phone-calls-migration-python).

We identify the following categories of tasks that need to be performed over KGs: Knowledge Graph Completion, Decision-Making, and Soft Searching.

### Knowledge Graph Completion

Here we term any task which creates new facts for the KG as *Knowledge Graph Completion*.

#### Relation Prediction (a.k.a. Link prediction)

We often want to find new connections in our Knowledge Graphs. Often, we need to understand how two concepts are connected. This is the case of **binary** Relation prediction, which all existing literature concerns itself with. Grakn is a [Hypergraph](https://en.wikipedia.org/wiki/Hypergraph), where Relations are [Hyperedges](https://en.wikipedia.org/wiki/Glossary_of_graph_theory_terms#hyperedge). Therefore, in general, the Relations we may want to predict may be **ternary** (3-way) or even **[N-ary](https://en.wikipedia.org/wiki/N-ary_group)** (N-way), which goes beyond the research we have seen in this domain.

When predicting Relations, there are several scenarios we may have. When predicting binary Relations between the members of one set and the members of another set, we may need to  predict them as:

- One-to-one

- One-to-many

- Many-to-many

*Examples:* The problem of predicting which disease(s) a patient has is a one-to-many problem. Whereas, predicting which drugs in the KG treat which diseases is a many-to-many problem.

Notice also that recommender systems are one use case of one-to-many binary Relation prediction.

We anticipate that solutions working well for the one-to-one case will also be applicable (at least to some extent) to the one-to-many case and cascade also to the many-to-many case.

***In KGLIB*** [*Knowledge Graph Convolutional Networks* (KGCNs)](https://github.com/graknlabs/kglib/tree/master/kglib/kgcn) performs Relation prediction using an approach based on [Graph Networks](https://github.com/deepmind/graph_nets) from DeepMind. This can be used to predict **binary**, **ternary**, or **N-ary** relations. This is well-supported for the one-to-one case and the one-to-many case.

#### Attribute Prediction

We would like to predict one or more Attributes of a Thing, which may include also prediction of whether that Attribute should even be present at all.

***In KGLIB*** [*Knowledge Graph Convolutional Networks* (KGCNs)](https://github.com/graknlabs/kglib/tree/master/kglib/kgcn) can be used to directly learn Attributes for any Thing. This requires some minor additional functionality to be added (we intend to build this imminently).

#### Subgraph Prediction

We can extend N-ary Relation and Attribute prediction to include Entity prediction, and in fact connected graphs of Entities, Relations, and Attributes as entire subgraphs. It may be possible to determine that such a graph is missing from an existing partially complete Knowledge Graph.

#### Building Concept Embeddings

Embeddings of Things and/or Types are universally useful for performing other downstream machine learning or data science tasks. Their usefulness comes in storing the context of a Concept in the graph as a numerical vector. 
These vectors are easy to ingest into other ML pipelines.
The benefit of building general-purpose embeddings is therefore to make use of them in multiple other pipelines. This reduces the expense of traversing the Knowledge Graph, since this task can be performed once and the output re-used more than once.

***In KGLIB*** [*Knowledge Graph Convolutional Networks* (KGCNs)](https://github.com/graknlabs/kglib/tree/master/kglib/kgcn) can be used to build general-purpose embeddings. This requires additional functionality, since a generic loss function is required in order to train the model in an unsupervised fashion. At its simplest, this can be achieved by measuring the shortest distance across the KG between two Things. This can be achieved trivially in Grakn using [`compute path`](https://dev.grakn.ai/docs/query/compute-query#compute-the-shortest-path).

#### Rule Mining (a.k.a. Association Rule Learning)

Known largely as [Association Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning) in the literature, here we refer to Horn Clause Rule Mining. The objective is to search the Knowledge Graph for new [Grakn Rules](https://dev.grakn.ai/docs/schema/rules) that may be applicable in the form of :
```graql
mined-rule sub rule,
when {
  [antecedent (the conditions)]
}, then {
  [consequent (the conclusions)]
};
```

We anticipate that the validity of these rules needs to be checked by hand, since once committed to the graph they are assumed to be correct and will be applied across the KG.

Rule mining is a very important field for KG completion. Finding and verifying rules can augment the existing knowledge at scale, since the Rule will be applied wherever the antecedent is found to be true.

We deem Rule mining a form of [inductive reasoning](https://en.wikipedia.org/wiki/Inductive_reasoning), as opposed to the [deductive reasoning](https://en.wikipedia.org/wiki/Deductive_reasoning) built in to Grakn (the method by which the induced Rules are applied).

#### Ontology Merging

Merging ontologies is a relatively common problem. Most often, users wish to merge their own proprietary Knowledge Graph with a public Knowledge Graph, for example [ConceptNet](http://conceptnet.io), [Gene Ontology (GO)](http://geneontology.org), [Disease Ontology (DO)](http://disease-ontology.org).

Grakn's highly flexible knowledge representation features means that this isn't challenging at all if the two ontologies contain non-overlapping Entities, even if the Entities of the two KGs are interrelated.

The challenge here is to find a mapping between the structure of the two KGs. If Types or Things (Type instances) overlap between the two KGs, then they need to be merged.

This decomposes the problem to that of matching between the two KGs. Grakn's schema helps with this task, since we can use this to perform matching between the structures of the two KGs, and thereby find a mapping between them. Matching of data can be framed as either link prediction, or a comparison of graph embeddings.

#### Automated Knowledge Graph Creation

Often, Grakn users want to build a KG (or a subgraph of a KG) from raw data sources. This could be CSVs, SQL databases, bodies of text or information crawled on the web. 

Data with tabular structure is best migrated to Grakn manually until this field is far more mature. This is a result of the fact that a human reader of structure can infer the meaning of the information, but with only a field name or column name to go on, automated processes will lack the context necessary to perform well. Combining data from many sources in this fashion is a core strength of Grakn, and is not hard to achieve, as per the [Grakn docs](https://dev.grakn.ai/docs/examples/phone-calls-migration-python).

Creating KGs from unstructured information such as text, however, is actually achievable for this task, thanks to the rich context in the data and the wide array of open-source NLP and NLU frameworks available to use. The output of these frameworks can be ingested into a KG with ease, and semi-autonomously used to build a KG of the domain described by the unstructured data. See [this post](https://blog.grakn.ai/text-mined-knowledge-graphs-beyond-text-mining-1ff207a7d850) for more details on how to go about this. For discussions on integrating NLP and NLU with Grakn, check out the #nlp channel on the [Grakn Slack](https://grakn.ai/slack).

### Decision-Making

Querying a compete knowledge graph may not be enough to inform complex of difficult decisions; we require methods specifically to help us find the right decision to make.

#### Expert Systems

Given a scenario, Expert Systems automatically make decisions or prompt the best course of action.

For many applications, Expert Systems can be built entirely with the features provided by Grakn out-of-the-box. They make extensive use of deductive reasoning by utilising Grakn's Rules, where data describing the scenario is added to the KG programmatically with `insert` statements (made via one of the [Grakn Clients](https://dev.grakn.ai/docs/client-api/overview)). `match` queries can then return the right course of action given the scenario.

***In KGLIB*** we need to create examples of Expert Systems, and outline any best practices for building them. This includes recommendations for when symbolic logic alone doesn't provide sufficient answers, and needs to be combined with ML.

#### Optimal Pattern Finding

In general, this problem is one of finding an optimal path between two Things in a KG. This generally means taking account of a cost of traversing edges and nodes. If the cost of all traversals are equal, then [`compute path`](https://dev.grakn.ai/docs/query/compute-query#compute-the-shortest-path) will already perform this task, which is one of Grakn's built-in distributed algorithms. Here, the subgraph that can be traversed can be constrained with the `in` keyword.

Beyond `compute path`, this problem concerns optimisation and is not necessarily best solved via ML methods; although it may be a candidate for an approach based on Reinforcement Learning.

#### System Design Automation and Configuration Automation

Typically this problem arises in engineering problems, most often in system design, where many systems need to be constructed and done so optimally according to the task they must fulfil and the constraints upon them.
This should be wholly or partially solvable with Grakn's automated reasoning. This task is particularly exciting, since solving it brings us closer to automating engineering design processes.

***In KGLIB*** we need to build examples of how to build such methodologies, along with best practice guidance.

### Soft Searching

#### Fuzzy Pattern Matching

[Graql](https://dev.grakn.ai/docs/query/overview) is a highly expressive language that we can use to query Grakn. Included in Graql is the ability to make ambiguous queries. In some cases however, we may want to retrieve a list of the best matches rather than an equally-weighted list of exact matches. This requires a solution that goes beyond Graql.

#### Querying and Responding via Natural Language

This problem reduces to converting between natural language and [Graql](https://dev.grakn.ai/docs/query/overview). Graql is expressive and closely resembles spoken English. However, there is still a gap between natural language and Graql. Finding a bridge for this gap will allow non-technical users to ask questions of the KG in natural language, and receive answers in natural language. 
This has wide applications, with particular rising interest in building chatbots for the web. At present, the most favourable solution architecture is to use a readily available NLU component, and translate the intentions that this component identifies into Graql.

### Summary

We see that the tasks that we want to perform over Knowledge Graphs are varied. Having at our disposal  a toolbox of methods to perform these tasks is a very exciting prospect! This would be enabling across so many industries.

This is the motivation for KGLIB. We would be delighted if you would help us to progress thiw work. You can help in so many ways:

- Starring the repo to let us know you like it ;)
- Telling us which tasks are important for you
- Highlighting tasks we have omitted
- Getting involved with the code and contributing to the solutions that interest you!

Thanks for reading!
