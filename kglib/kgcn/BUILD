load("@io_bazel_rules_python//python:python.bzl", "py_library")


py_library(
    name = "kgcn",
    deps = [
        '//kglib/kgcn/examples/diagnosis',
        '//kglib/kgcn/learn',
        '//kglib/kgcn/models',
        '//kglib/kgcn/pipeline',
        '//kglib/kgcn/plot',
    ],
    visibility=['//visibility:public']
)