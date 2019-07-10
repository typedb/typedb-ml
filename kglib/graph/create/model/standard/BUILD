load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "convert_test",
    srcs = [
        "convert_test.py"
    ],
    deps = [
        "standard",
        "//kglib/graph/utils/test"
    ]
)

py_library(
    name = "standard",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn',
        '//kglib/graph/create/model/common'
    ],
    visibility=['//visibility:public']
)