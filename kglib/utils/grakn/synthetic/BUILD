load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")


py_library(
    name = "synthetic",
    deps = [
        '//kglib/utils/grakn/synthetic/statistics',
        '//kglib/utils/grakn/synthetic/examples',
    ],
    visibility=['//visibility:public']
)