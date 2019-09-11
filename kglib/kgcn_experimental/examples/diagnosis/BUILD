load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "diagnosis_test",
    srcs = [
        "diagnosis_test.py"
    ],
    deps = [
        "diagnosis",
    ]
)

py_test(
    name = "diagnosis_IT",
    srcs = [
        "diagnosis_IT.py"
    ],
    deps = [
        "diagnosis",
        "//kglib/utils/grakn/test",
    ],
    data = [
        "@graknlabs_grakn_core//:assemble-mac-zip",
#        ":test_data/schema.gql"
    ]
)

py_library(
    name = "diagnosis",
    srcs = glob(['**/*.py']),
    deps = [
        "//kglib/kgcn_experimental/pipeline",
        "//kglib/kgcn_experimental/learn",
        "//kglib/kgcn_experimental/plot",
        "//kglib/kgcn_experimental/models",
        "//kglib/utils/grakn/synthetic",
        "@graknlabs_client_python//:client_python",
        requirement('graph_nets'),
        requirement('numpy'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt'),
    ],
    visibility=['//visibility:public']
)