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
        # Networkx deps
        requirement('networkx'),
        requirement('decorator'),

        # Graph nets deps
        requirement('absl-py'),
        requirement('cloudpickle'),
        requirement('contextlib2'),
        # requirement('decorator'),
        requirement('dm-sonnet'),
        requirement('future'),
        requirement('graph-nets'),
        # requirement('networkx'),
        requirement('numpy'),
        requirement('semantic-version'),
        requirement('six'),
        requirement('tensorflow-probability'),
        requirement('wrapt'),

        # Tensorflow deps
        # requirement('absl-py'),
        requirement('astor'),
        requirement('gast'),
        requirement('google-pasta'),
        requirement('keras-applications'),
        requirement('keras-preprocessing'),
        # requirement('six'),
        requirement('protobuf'),
        requirement('tensorboard'),
        requirement('tensorflow'),
        requirement('tensorflow-estimator'),
        requirement('termcolor'),
        # requirement('wrapt'),
    ],
    visibility=['//visibility:public']
)