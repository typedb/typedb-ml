load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "ordered_test",
    srcs = [
        "kgcn/core/ingest/traverse/data/sample/ordered_test.py"
    ],
    deps = [
        "kglib"
    ],
)

py_test(
    name = "random_sampling_test",
    srcs = [
        "kgcn/core/ingest/traverse/data/sample/random_sampling_test.py"
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
    name = "builder_test",
    srcs = [
        "kgcn/core/ingest/traverse/data/builder_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "neighbour_test",
    srcs = [
        "kgcn/core/ingest/traverse/data/neighbour_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "schema_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "kgcn/core/ingest/traverse/schema/traversal_test.py"
    ],
    deps = [
        "kglib",
    ]
)

py_test(
    name = "context_array_test",
    srcs = [
        "kgcn/core/ingest/preprocess/context_array_test.py"
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