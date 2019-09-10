load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "query_graph_test",
    srcs = [
        "query_graph_test.py"
    ],
    deps = [
        "query",
    ],
)

py_library(
    name = "query",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
    ],
    visibility=['//visibility:public']
)