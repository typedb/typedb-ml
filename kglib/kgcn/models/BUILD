load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "attribute_test",
    srcs = [
        "attribute_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "attribute_IT",
    srcs = [
        "attribute_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_test(
    name = "embedding_test",
    srcs = [
        "embedding_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "typewise_test",
    srcs = [
        "typewise_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "typewise_IT",
    srcs = [
        "typewise_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_test(
    name = "core_IT",
    srcs = [
        "core_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_library(
    name = "models",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('graph_nets'),
        requirement('numpy'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),

        # Tensorflow deps
        requirement('absl-py'),
        requirement('astor'),
        requirement('gast'),
        requirement('google_pasta'),
        requirement('keras-applications'),
        requirement('keras-preprocessing'),
#        requirement('numpy'),
        requirement('opt-einsum'),
        requirement('six'),
        requirement('protobuf'),
        requirement('tensorboard'),
        requirement('tensorflow-estimator'),
        requirement('termcolor'),
        requirement('wrapt'),

    ],
    visibility=['//visibility:public']
)