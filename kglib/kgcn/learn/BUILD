load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "metrics_test",
    srcs = [
        "metrics_test.py"
    ],
    deps = [
        "learn"
    ]
)

py_test(
    name = "learn_IT",
    srcs = [
        "learn_IT.py"
    ],
    deps = [
        "learn",
        "//kglib/kgcn/models",
        "//kglib/kgcn/plot"
    ]
)

py_library(
    name = "learn",
    srcs = [
        'feed.py',
        'learn.py',
        'loss.py',
        'metrics.py',
    ],
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

        # Scipy deps
        requirement('scipy')

    ],
    visibility=['//visibility:public']
)