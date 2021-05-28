load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "test",
    srcs = [
        'case.py',
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
