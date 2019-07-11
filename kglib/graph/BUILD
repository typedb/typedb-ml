load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "graph",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn:kgcn',
        '//kglib/graph/label:label',
        '//kglib/graph/create:create',
        '//kglib/graph/create/model:model',
        '//kglib/graph/create/model/standard:standard',
        '//kglib/graph/create/model/math:math',
        '//kglib/graph/create/model/common:common',
    ],
    visibility=['//visibility:public']
)