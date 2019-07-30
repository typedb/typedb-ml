load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "pmf_test",
    srcs = [
        "pmf_test.py"
    ],
    deps = [
        "generate"
    ]
)

py_library(
    name = "generate",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('numpy'),
        requirement('pandas'),
        requirement('pytz'),
        requirement('python-dateutil')
    ],
    visibility=['//visibility:public']
)