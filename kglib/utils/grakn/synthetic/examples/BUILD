load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

exports_files(["diagnosis/schema.gql"])

py_library(
    name = "examples",
    srcs = glob(['**/*.py']),
    deps = [
        "@graknlabs_client_python//:client_python",
        requirement('numpy'),
    ],
    visibility=['//visibility:public']
)