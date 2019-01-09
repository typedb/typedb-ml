

py_test(
    name = "my_test",
    srcs = [
        "kgcn/my_test.py"
    ],
    deps = [
    ]
)

py_test(
    name = "ordered_test",
    srcs = [
        "kgcn/src/neighbourhood/data/sampling/ordered_test.py"
    ],
    deps = [
        "kgcn"
    ]
)

py_test(
    name = "random_test",
    srcs = [
        "kgcn/src/neighbourhood/data/sampling/random_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Takes ages to run in bazel
py_test(
    name = "label_extraction_test",
    srcs = [
        "kgcn/src/use_cases/attribute_prediction/label_extraction_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "metrics_test",
    srcs = [
        "kgcn/src/models/metrics_test.py"
    ],
    deps = [
        "kgcn",
    ]
)


py_library(
    name = "kgcn",
    srcs = [
        'kgcn/encoder/tmp/TestShit.py',
        'kgcn/neighbourhood/__init__.py',
        'kgcn/neighbourhood/schema/__init__.py',
        'kgcn/my_test.py',
        'kgcn/preprocess/__init__.py',
        'kgcn/preprocess/tmp/__init__.py',
        'kgcn/preprocess/tmp/preprocess_test.py',
        'kgcn/models/tmp/manager_test.py',
        'kgcn/models/tmp/model_old.py',
        'kgcn/models/tmp/downstream_old.py',
        'kgcn/models/tmp/usage.py',
        'kgcn/models/tmp/manager.py',
        'kgcn/use_cases/__init__.py',
        'kgcn/use_cases/attribute_prediction/__init__.py',
        'kgcn/tmp/main.py',
        'kgcn/src/paper/trial_superset_description.py',
        'kgcn/src/encoder/encode.py',
        'kgcn/src/encoder/schema_test.py',
        'kgcn/src/encoder/tf_hub.py',
        'kgcn/src/encoder/boolean.py',
        'kgcn/src/encoder/encode_test.py',
        'kgcn/src/encoder/tf_hub_test.py',
        'kgcn/src/encoder/boolean_test.py',
        'kgcn/src/encoder/schema.py',
        'kgcn/src/neighbourhood/schema/traversal.py',
        'kgcn/src/neighbourhood/schema/strategy.py',
        'kgcn/src/neighbourhood/schema/traversal_test.py',
        'kgcn/src/neighbourhood/schema/executor.py',
        'kgcn/src/neighbourhood/data/executor_test.py',
        'kgcn/src/neighbourhood/data/traversal.py',
        'kgcn/src/neighbourhood/data/sampling/random_test.py',
        'kgcn/src/neighbourhood/data/sampling/random_sampling.py',
        'kgcn/src/neighbourhood/data/sampling/ordered_test.py',
        'kgcn/src/neighbourhood/data/sampling/ordered.py',
        'kgcn/src/neighbourhood/data/sampling/sampler.py',
        'kgcn/src/neighbourhood/data/traversal_mocks.py',
        'kgcn/src/neighbourhood/data/traversal_test.py',
        'kgcn/src/neighbourhood/data/utils.py',
        'kgcn/src/neighbourhood/data/executor.py',
        'kgcn/src/preprocess/date_to_unixtime.py',
        'kgcn/src/preprocess/preprocess.py',
        'kgcn/src/preprocess/raw_array_builder_test.py',
        'kgcn/src/preprocess/raw_array_builder.py',
        'kgcn/src/models/metrics.py',
        'kgcn/src/models/manager_test.py',
        'kgcn/src/models/model_test.py',
        'kgcn/src/models/aggregation.py',
        'kgcn/src/models/metrics_test.py',
        'kgcn/src/models/model.py',
        'kgcn/src/models/learners.py',
        'kgcn/src/models/learners_test.py',
        'kgcn/src/models/manager.py',
        'kgcn/src/models/initialise.py',
        'kgcn/src/examples/toy/main.py',
        'kgcn/src/examples/animal_trade/persistence.py',
        'kgcn/src/examples/animal_trade/main.py',
        'kgcn/src/use_cases/attribute_prediction/label_extraction_test.py',
        'kgcn/src/use_cases/attribute_prediction/label_extraction.py',
        ]
)