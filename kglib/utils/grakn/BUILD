load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "grakn",
    deps = [
        '//kglib/utils/grakn/test',
        '//kglib/utils/grakn/object',
        '//kglib/utils/grakn/synthetic',
        '//kglib/utils/grakn/type',
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
