load("@rules_python//python:defs.bzl", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_library(
    name = "synthetic",
    deps = [
        '//kglib/utils/grakn/synthetic/statistics',
        '//kglib/utils/grakn/synthetic/examples',
    ],
    visibility=['//visibility:public']
)