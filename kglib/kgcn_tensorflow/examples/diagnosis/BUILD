load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

py_test(
    name = "diagnosis_test",
    srcs = [
        "diagnosis_test.py"
    ],
    deps = [
        "diagnosis",
        "//kglib/utils/graph/test",
        graknlabs_kglib_requirement('numpy'),
        graknlabs_kglib_requirement('networkx'),
        graknlabs_kglib_requirement('decorator'),
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