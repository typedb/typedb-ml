load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

exports_files(["diagnosis/schema.gql"])

py_library(
    name = "examples",
    srcs = [
        'diagnosis/generate.py',
    ],
    deps = [
        "@graknlabs_client_python//:client_python",
        requirement('numpy'),
    ],
    visibility=['//visibility:public']
)