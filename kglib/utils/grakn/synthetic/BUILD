load("@io_bazel_rules_python//python:python.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "synthetic",
    srcs = glob(['**/*.py']),
    deps = [
        '//kglib/utils/grakn/synthetic/statistics:statistics',
        requirement('numpy'),
        requirement('pandas'),
        requirement('pytz'),
        requirement('python-dateutil'),
        requirement('six')
    ],
    visibility=['//visibility:public']
)