load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "plotting_test",
    srcs = [
        "plotting_test.py"
    ],
    deps = [
        "plot"
    ]
)

py_library(
    name = "plot",
    srcs = glob(['**/*.py']),
    deps = [
        # Matplotlib deps
        requirement('cycler'),
        requirement('kiwisolver'),
        requirement('matplotlib'),
        requirement('pyparsing'),
        requirement('python-dateutil'),

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