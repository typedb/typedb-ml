load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "grakn",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/utils/grakn/test:test',
        '//kglib/utils/grakn/object:object',
        '//kglib/utils/grakn/synthetic:synthetic',
    ],
    visibility=['//visibility:public']
)