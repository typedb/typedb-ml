#load("@rules_python//python:defs.bzl", "py_test", "py_library")
#load("@vaticle_kglib_pip//:requirements.bzl",
#       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")
#
#py_test(
#    name = "encode_test",
#    srcs = [
#        "encode_test.py"
#    ],
#    deps = [
#        "pipeline"
#    ]
#)
#
#py_test(
#    name = "utils_test",
#    srcs = [
#        "utils_test.py"
#    ],
#    deps = [
#        "pipeline",
#        "//kglib/utils/graph/test"
#    ]
#)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
        "**/*"
    ]),
    license_type = "apache",
)
