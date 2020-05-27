load("@rules_python//python:defs.bzl", "py_library")


py_library(
    name = "graph",
    srcs = glob(['**/*.py']),
    deps = [
        '//kglib/utils/graph/query',
        '//kglib/utils/graph/thing',
    ],
    visibility=['//visibility:public']
)