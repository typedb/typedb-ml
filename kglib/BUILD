load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "ordered_test",
    srcs = [
        "kgcn/core/ingest/neighbourhood/data/sampling/ordered_test.py"
    ],
    deps = [
        "kglib"
    ],
)

py_test(
    name = "random_sampling_test",
    srcs = [
        "kgcn/core/ingest/neighbourhood/data/sampling/random_sampling_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "label_extraction_test",
    srcs = [
        "kgcn/learn/attribute/label_extraction_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "report_test",
    srcs = [
        "kgcn/learn/metrics/report_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "tf_hub_test",
    srcs = [
        "kgcn/core/ingest/encode/tf_hub_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "schema_test",
    srcs = [
        "kgcn/core/ingest/encode/schema_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "encode_test",
    srcs = [
        "kgcn/core/ingest/encode/encode_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "boolean_test",
    srcs = [
        "kgcn/core/ingest/encode/boolean_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "data_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "kgcn/core/ingest/neighbourhood/data/traversal_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "data_executor_test",
    main = "executor_test.py",
    srcs = [
        "kgcn/core/ingest/neighbourhood/data/executor_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "schema_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "kgcn/core/ingest/neighbourhood/schema/traversal_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "raw_array_builder_test",
    srcs = [
        "kgcn/core/ingest/preprocess/raw_array_builder_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "model_test",
    srcs = [
        "kgcn/core/model_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_library(
    name = "kglib",
    srcs = glob(['__init__.py', 'kgcn/**/*.py']),
    deps = [
        # Grakn deps
        requirement('grakn'),
        requirement('grpcio'),

        # TensorFlow deps
        requirement('tensorflow'),
        requirement('numpy'),
        requirement('protobuf'),
        requirement('six'),
        requirement('absl-py'),
        requirement('keras_applications'),
        requirement('keras_preprocessing'),
        requirement('gast'),
        requirement('astor'),
        requirement('termcolor'),

        requirement('tensorflow-hub'),
        requirement('scikit-learn'),
        requirement('scipy'),
    ],
    visibility=['//visibility:public']
)