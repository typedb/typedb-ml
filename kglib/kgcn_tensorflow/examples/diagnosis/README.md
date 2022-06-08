# KGCN Diagnosis Example

This example is entirely fabricated as a demonstration for how to construct a KGCN pipeline. Since the data for this example is generated synthetically, it also functions as a test platform for the KGCN model.

Studying the schema for this example (using TypeDB Workbase's Schema Designer), we have people who present symptoms, with some severity. Separately, we may know that certain symptoms can be caused by a disease. We also know information that contributes to risk-factors for certain diseases. These risk factors are determined by rules defined in the schema. Lastly, people can be diagnosed with a disease.

![Diagnosis Schema](.images/diagnosis_schema.png)

## Running the Example

Once you have [installed KGLIB via pip](../../#getting-started---running-the-machine-learning-pipeline) you can run the example as follows:

1. Make sure a TypeDB server is running

2. Run the example: `python -m kglib.kgcn_tensorflow.examples.diagnosis.diagnosis`

   The database, schema and seed data will be created automatically. Data is generated synthetically. The whole example should complete in under 10 minutes

3. You should observe console output to indicate that the pipeline is running and that the model is learning. Afterwards two plots should be created to visualise the training process and examples of the predictions made.

## Diagnosis Pipeline

The process conducted by the example is as follows:

1. Generate synthetic graphs, each graph is used as an *example*
   - This requires specifying queries that will retrieve Concepts from TypeDB
   - The answers from these queries are used to create subgraphs, stored in-memory as networkx graphs
2. Find the Types and Roles present in the schema. If any are not needed for learning then they should be excluded from the exhaustive list for better accuracy.
3. Run the pipeline
4. Write the predictions made to TypeDB

## Relation Prediction

The learner predicts three classes for each graph element. These are:

```
[
Element already existed in the graph (we wish to ignore these elements),
Element does not exist in the graph,
Element does exist in the graph
]
```

In this way we perform relation prediction by proposing negative candidate relations (TypeDB's rules help us with this). Then we train the learner to classify these negative candidates as **does not exist** and the correct relations as **does exist**.

## Results Output

### Console Reporting

During training, the console will output metrics for the performance on the training and test sets.

You should see output such as this for the diagnosis example:

```
# (iteration number), T (elapsed seconds), Ltr (training loss), Lge (test/generalization loss), Ctr (training fraction nodes/edges labeled correctly), Str (training fraction examples solved correctly), Cge (test/generalization fraction nodes/edges labeled correctly), Sge (test/generalization fraction examples solved correctly)
# 00000, T 4.4, Ltr 0.7928, Lge 0.7518, Ctr 0.4900, Str 0.0000, Cge 0.5000, Sge 0.0000
# 00020, T 9.8, Ltr 0.7036, Lge 0.6957, Ctr 0.5100, Str 0.0200, Cge 0.5000, Sge 0.0000
# 00040, T 12.1, Ltr 0.5384, Lge 0.4540, Ctr 0.7900, Str 0.6100, Cge 0.8100, Sge 0.6300
# 00060, T 14.4, Ltr 0.7434, Lge 0.3631, Ctr 0.7650, Str 0.5400, Cge 0.8850, Sge 0.7900
# 00080, T 16.7, Ltr 0.3643, Lge 0.2464, Ctr 0.9200, Str 0.8800, Cge 0.9350, Sge 0.8900
# 00100, T 19.0, Ltr 0.2806, Lge 0.1590, Ctr 0.9600, Str 0.9600, Cge 0.9650, Sge 0.9500
# 00120, T 21.3, Ltr 0.5488, Lge 0.2577, Ctr 0.9100, Str 0.8400, Cge 0.9300, Sge 0.8800
# 00140, T 23.5, Ltr 0.2913, Lge 0.2590, Ctr 0.9650, Str 0.9600, Cge 0.9200, Sge 0.8600
# 00160, T 25.8, Ltr 0.2603, Lge 0.1476, Ctr 0.9650, Str 0.9600, Cge 0.9700, Sge 0.9600
# 00180, T 28.1, Ltr 0.2656, Lge 0.1411, Ctr 0.9650, Str 0.9600, Cge 0.9700, Sge 0.9600
...
```

Take note of the key:

- \# - iteration number
- T - elapsed seconds
- Ltr - training loss
- Lge - test/generalization loss
- Ctr - training fraction nodes/edges labeled correctly
- Str - training fraction examples solved correctly
- Cge - test/generalization fraction nodes/edges labeled correctly
- Sge - test/generalization fraction examples solved correctly

The element we are most interested in is `Sge`, the proportion of subgraphs where all elements of the subgraph were classified correctly. This therefore represents an entirely correctly predicted example.

### Diagrams

#### Training Metrics

Upon running the example you will also get plots from matplotlib saved to your working directory.

You will see plots of metrics for the training process (training iteration on the x-axis) for the training set (solid line), and test set (dotted line). From left to right:

- The absolute loss across all of the elements in the dataset
- The fraction of all graph elements predicted correctly across the dataset
- The fraction of completely solved examples (subgraphs extracted from TypeDB that are solved in full)

![learning metrics](.images/learning.png)

#### Visualise the Predictions

We also receive a plot of some of the predictions made on the test set. 

![predictions made on test set](.images/graph.png)

**Blue box:** Ground Truth 

- Preexisting (known) graph elements are shown in blue

- Relations and role edges that **should be predicted to exist** are shown in green

- Candidate relations and role edges that **should not be predicted to exist** are shown faintly in red

**Black boxes**: Model Predictions at certain message-passing steps

This uses the same colour scheme as above, but opacity indicates a probability given by the model.

These boxes shows the score assigned to the class **does exist**.

Therefore, for good predictions we want to see no blue elements, and for the red elements to fade out as more messages are passed, the green elements becoming more certain.

## How does Link Prediction work?

The methodology used for Relation prediction is as follows:

In this example, we aim to predict `diagnosis` Relations. We have the correct `diagnosis` relations, and we write a TypeDB rule to insert `candidate-diagnosis` relations as negative targets. They are added wherever a real `diagnosis` Relation could logically exist, but does not.

We then teach the KGCN to distinguish between the positive and negative targets.

## Querying for the Train/Test Datasets

We do this by creating *examples*, where each example is a subgraph extracted from a TypeDB knowledge Graph. These subgraphs contain positive and negative instances of the relation to be predicted.

A single subgraph is created by making multiple queries to TypeDB. In this example, each subgraph centres around a `person` who is uniquely identifiable. This is important, since we want the results for these queries to return information about the vacinity of an individual. That is, we want information about a subgraph rather than the whole graph. For this example you can find the queries made in [diagnosis.py](diagnosis.py).

A single subgraph is extracted from TypeDB by making these queries and combining the results into a graph. For your own domain you should find queries that will retrieve the most relevant information for the Relations you are trying to predict.

We can visualise such a subgraph by running these queries one after the other in TypeDB Workbase:

![queried subgraph](.images/queried_subgraph.png)

You can get the relevant version of TypeDB Workbase from the Assets of the [latest Workbase release](https://github.com/vaticle/workbase/releases/latest).

Using Workbase like this is a great way to understand the subgraphs that are actually being delivered to the KGCN -- a great understanding and debugging tool.

## Modifying the Example

If you need to customise the learning or model used for your own use case, you'll need to make changes to the [pipeline](https://github.com/vaticle/kglib/tree/master/kglib/kgcn/pipeline/pipeline.py) used.

Consider tuning parameters and adjusting elements of the pipeline if you need to improve the accuracy that you see. Start by adjusting `num_processing_steps_tr`, `num_processing_steps_ge`, `num_training_iterations`.