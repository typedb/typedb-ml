load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@graknlabs_bazel_distribution_pip//:requirements.bzl", deployment_requirement = "requirement")

py_library(
    name = "kglib",
    srcs = [
        '__init__.py',
    ],
    deps = [
        '//kglib/kgcn',
        '//kglib/utils',
    ],
    visibility=['//visibility:public']
)
