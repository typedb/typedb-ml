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
        "//kglib/utils/grakn/synthetic/examples:diagnosis/schema.gql"
    ]
)

py_library(
    name = "diagnosis",
    srcs = glob(['**/*.py']),
    deps = [
        "//kglib/kgcn/pipeline",
        "//kglib/kgcn/learn",
        "//kglib/kgcn/plot",
        "//kglib/kgcn/models",
        "//kglib/utils/grakn/synthetic",
        "@graknlabs_client_python//:client_python",
    ],
    visibility=['//visibility:public']
)