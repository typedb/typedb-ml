load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@vaticle_bazel_distribution_pip//:requirements.bzl", deployment_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")


py_library(
    name = "kglib",
    srcs = [
        '__init__.py',
    ],
    deps = [
        '//kglib/kgcn_tensorflow:kgcn',
        '//kglib/utils',
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)