load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

py_test(
    name = "encode_test",
    srcs = [
        "encode_test.py"
    ],
    deps = [
        "pipeline"
    ]
)

py_test(
    name = "utils_test",
    srcs = [
        "utils_test.py"
    ],
    deps = [
        "pipeline",
        "//kglib/utils/graph/test"
    ]
)


py_library(
    name = "pipeline",
    srcs = [
        'pipeline.py',
    ],
    deps = [
        vaticle_kglib_requirement('graph_nets'),
        vaticle_kglib_requirement('numpy'),
        vaticle_kglib_requirement('dm-sonnet'),
        vaticle_kglib_requirement('tensorflow'),
        vaticle_kglib_requirement('tensorflow-probability'),
        vaticle_kglib_requirement('semantic-version'),
        vaticle_kglib_requirement('contextlib2'),
        vaticle_kglib_requirement('wrapt'),
        "//kglib/utils/graph",
    ],
    visibility=['//visibility:public']
)