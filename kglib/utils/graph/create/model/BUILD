load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "model",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn',
        '//kglib/graph/create/model/common',
        '//kglib/graph/create/model/math',
        '//kglib/graph/create/model/standard'
    ],
    visibility=['//visibility:public']
)