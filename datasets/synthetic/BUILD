load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "synthetic_graphs",
    srcs = glob(['**/*.py']),
    deps = [
        '//datasets/synthetic/generate:generate',
        requirement('numpy'),
        requirement('pandas'),
        requirement('pytz'),
        requirement('python-dateutil')
    ],
    visibility=['//visibility:public']
)