load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

filegroup(
    name = "graql-schema",
    srcs = [
        "diagnosis/schema.gql",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "examples",
    srcs = [
        'diagnosis/generate.py',
    ],
    deps = [
        "@vaticle_typedb_client_python//:client_python",
        vaticle_kglib_requirement('numpy'),
    ],
    visibility=['//visibility:public']
)