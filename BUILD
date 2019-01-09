

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

# Bazel shows one test passes, should be 3
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

py_test(
    name = "tf_hub_test",
    srcs = [
        "kgcn/src/encoder/tf_hub_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "schema_test",
    srcs = [
        "kgcn/src/encoder/schema_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Shouldn't pass, but does
py_test(
    name = "encode_test",
    srcs = [
        "kgcn/src/encoder/encode_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Throws errors trying to run tests
py_test(
    name = "data_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "kgcn/src/neighbourhood/data/traversal_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Throws errors trying to run tests
py_test(
    name = "data_executor_test",
    main = "executor_test.py",
    srcs = [
        "kgcn/src/neighbourhood/data/executor_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Tests should fail
py_test(
    name = "schema_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "kgcn/src/neighbourhood/schema/traversal_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

# Bazel shows 1 test passes, but there should be 30/31 passes
py_test(
    name = "raw_array_builder_test",
    srcs = [
        "kgcn/src/preprocess/raw_array_builder_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_library(
    name = "kgcn",
    srcs = glob(['kgcn/**/*.py'])
)