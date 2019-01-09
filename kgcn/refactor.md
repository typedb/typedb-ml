
1. Customise the learning parameters, in some clear way, including:
    - numerical parameters 
    - strategies
    - encoders
    
*Pre-TensorFlow model*
These components are used repeatedly for each neighbour hop, but could require different parameters each
neighbour sampling params: 
    query limits, 
    sampling nature e.g. ordered, pseudo-random, random

*Within Tensorflow model*
Name scoping?

encoding parameters
    encoder to user per-type

normalisation parameters(?)

These components are used repeatedly for each neighbour hop, but could require different parameters each
    Aggregator parameters
        Weight initialiser
        Bias
        Weight regulariser
        Activation  
        Dropout
        Layer Type (currently dense)
        Reduction method
    Combination parameters
        Weight initialiser
        Weight regularisers
    
    Normaliser parameters

Loss method

kgcn = KGCN(traversal_params={}, aggregation_params={'bias': False}, combination_params={})
Any arguments provided here should override the default dict params




2. Direct supervised learning for:
    - Unknown downstream learning (arbitrary user pipeline), with ready-made components:
        - Attribute prediction
        - Link prediction

3. Generate unsupervised embeddings, subsequently perform either:
    - Unknown downstream learning (arbitrary user pipeline), with ready-made components:
        - Attribute prediction
        - Link prediction

4. Visualise the model/learning in TensorBoard

5. Save the traversal output arrays to file in order to quickly iterate on the learning model

6. Advanced: Customise the neural net design

7. Low priority: Support learning outside TensorFlow, using other libraries etc




2. + 3.

traverser = Traverser(params)

traversals = traverser.traverse(concepts)

# Do any saving/loading of traversals and labels to/from file

embedder = Embedder(params)  # This is agnostic to training, evaluation, prediction etc

# Get the output tensors, e.g. embeddings, summary writers, initialisers etc
output_tensors = embedder.build()

classifier = SimpleMultiClassClassifier()

kgcn = KGCN(params)

# Create embeddings tensor
embeddings = kgcn.get_embeddings()

udl = UnknownDownstreamLearning(embeddings)
udl.train(concepts, labels, grakn_connection)


kgcn = SupervisedKGCNClassifier(params)
kgcn = KGCNMultiClassClassifier(Embedder(params), classifier_params)
kgcn = KGCN(params)
train_results = kgcn.train(concepts, labels)
eval_results = kgcn.eval(concepts, labels)
predictions = kgcn.predict(concepts, labels)










