load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "kgcn",
    deps = [
        '//kglib/kgcn_tensorflow/examples/diagnosis',
        '//kglib/kgcn_tensorflow/learn',
        '//kglib/kgcn_tensorflow/models',
        '//kglib/kgcn_tensorflow/pipeline',
        '//kglib/kgcn_tensorflow/plot',
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
