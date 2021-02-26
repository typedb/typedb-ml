load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

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
        graknlabs_kglib_requirement('graph_nets'),
        graknlabs_kglib_requirement('numpy'),
        graknlabs_kglib_requirement('dm-sonnet'),
        graknlabs_kglib_requirement('tensorflow'),
        graknlabs_kglib_requirement('tensorflow-probability'),
        graknlabs_kglib_requirement('semantic-version'),
        graknlabs_kglib_requirement('contextlib2'),
        graknlabs_kglib_requirement('wrapt'),
        "//kglib/utils/graph",
    ],
    visibility=['//visibility:public']
)