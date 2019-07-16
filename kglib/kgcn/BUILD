load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "ordered_test",
    srcs = [
        "core/ingest/traverse/data/sample/ordered_test.py"
    ],
    deps = [
        "kgcn"
    ],
)

py_test(
    name = "random_sampling_test",
    srcs = [
        "core/ingest/traverse/data/sample/random_sampling_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "label_extraction_test",
    srcs = [
        "learn/attribute/label_extraction_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "report_test",
    srcs = [
        "learn/metrics/report_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "tf_hub_test",
    srcs = [
        "core/ingest/encode/tf_hub_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "schema_test",
    srcs = [
        "core/ingest/encode/schema_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "encode_test",
    srcs = [
        "core/ingest/encode/encode_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "boolean_test",
    srcs = [
        "core/ingest/encode/boolean_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "builder_test",
    srcs = [
        "core/ingest/traverse/data/context/builder_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "builder_IT",
    srcs = [
        "core/ingest/traverse/data/context/builder_IT.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "neighbour_test",
    srcs = [
        "core/ingest/traverse/data/context/neighbour_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "schema_traversal_test",
    main = "traversal_test.py",
    srcs = [
        "core/ingest/traverse/schema/traversal_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "array_test",
    srcs = [
        "core/ingest/traverse/data/context/array_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "array_IT",
    srcs = [
        "core/ingest/traverse/data/context/array_IT.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "model_test",
    srcs = [
        "core/model_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_library(
    name = "kgcn",
    srcs = glob(['**/*.py']),
    deps = [
        # Grakn deps
        "@graknlabs_client_python//:client_python",

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