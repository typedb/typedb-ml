load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "graph",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn:kgcn',
        '//kglib/utils/graph/create:create',
    ],
    visibility=['//visibility:public']
)