load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "utils",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/utils/grakn',
        '//kglib/utils/graph',
        '//kglib/utils/test'
    ],
    visibility=['//visibility:public']
)