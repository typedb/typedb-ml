load("@io_bazel_rules_python//python:python.bzl", "py_library")

py_library(
    name = "kglib",
    srcs = glob(['__init__.py', 'kgcn/**/*.py']),
    deps = [
        '//kglib/kgcn',
        '//kglib/graph',
    ],
    visibility=['//visibility:public']
)