load("@io_bazel_rules_python//python:python.bzl", "py_library")


py_library(
    name = "graph",
    srcs = glob(['**/*.py']),
    deps = [
        '//kglib/utils/graph/query',
        '//kglib/utils/graph/thing',
    ],
    visibility=['//visibility:public']
)