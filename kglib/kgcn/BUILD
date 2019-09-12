load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "kgcn",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        requirement('graph_nets'),
        requirement('numpy'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt'),
        '//kglib/kgcn/examples/diagnosis',
        '//kglib/kgcn/learn',
        '//kglib/kgcn/models',
        '//kglib/kgcn/pipeline',
        '//kglib/kgcn/plot',
    ],
    visibility=['//visibility:public']
)