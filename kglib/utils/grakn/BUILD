load("@rules_python//python:defs.bzl", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")


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