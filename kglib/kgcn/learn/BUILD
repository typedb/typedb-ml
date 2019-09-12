load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
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
        "//kglib/kgcn_experimental/models",
        "//kglib/kgcn_experimental/plot"
    ]
)

py_library(
    name = "learn",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('graph_nets'),
        requirement('numpy'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt')

    ],
    visibility=['//visibility:public']
)