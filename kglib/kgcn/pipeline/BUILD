load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

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
        'encode.py',
        'pipeline.py',
        'utils.py',
    ],
    deps = [
        requirement('graph_nets'),
        requirement('numpy'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt'),
        "//kglib/utils/graph",
    ],
    visibility=['//visibility:public']
)