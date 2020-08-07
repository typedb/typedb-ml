load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

exports_files(["diagnosis/schema.gql"])

py_library(
    name = "examples",
    srcs = [
        'diagnosis/generate.py',
    ],
    deps = [
        "@graknlabs_client_python//:client_python",
        graknlabs_kglib_requirement('numpy'),
    ],
    visibility=['//visibility:public']
)