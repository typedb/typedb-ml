load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "label_test",
    srcs = [
        "label_test.py"
    ],
    deps = [
        "label",
    ]
)

py_library(
    name = "label",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn',
    ],
    visibility=['//visibility:public']
)