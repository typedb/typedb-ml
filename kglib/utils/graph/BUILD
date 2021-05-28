load("@rules_python//python:defs.bzl", "py_library")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "graph",
    srcs = glob(['**/*.py']),
    deps = [
        '//kglib/utils/graph/query',
        '//kglib/utils/graph/thing',
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
