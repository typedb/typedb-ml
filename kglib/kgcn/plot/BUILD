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
        requirement('networkx'),
        requirement('decorator'),
        requirement('numpy'),
        requirement('matplotlib'),
        requirement('pyparsing'),
        requirement('cycler'),
        requirement('python-dateutil'),
        requirement('kiwisolver'),
        requirement('graph_nets'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('dm-sonnet'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt')

    ],
    visibility=['//visibility:public']
)