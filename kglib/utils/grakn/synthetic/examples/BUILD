load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

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

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
        "diagnosis/*"
    ]),
    license_type = "apache",
)
