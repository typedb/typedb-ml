load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "utils",
    deps = [
        '//kglib/utils/grakn',
        '//kglib/utils/graph',
        '//kglib/utils/test'
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
