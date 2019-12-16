load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "diagnosis_test",
    srcs = [
        "diagnosis_test.py"
    ],
    deps = [
        "diagnosis",
        "//kglib/utils/graph/test",
        requirement('numpy'),
        requirement('networkx'),
        requirement('decorator'),
    ]
)


py_library(
    name = "diagnosis",
    srcs = [
        'diagnosis.py'
    ],
    deps = [
        "//kglib/kgcn/pipeline",
        "//kglib/kgcn/learn",
        "//kglib/kgcn/plot",
        "//kglib/kgcn/models",
        "//kglib/utils/grakn/synthetic",
        "//kglib/utils/grakn/type",
        "@graknlabs_client_python//:client_python",
    ],
    visibility=['//visibility:public']
)