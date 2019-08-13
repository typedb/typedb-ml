load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "from_queries_test",
    srcs = [
        "from_queries_test.py"
    ],
    deps = [
        "create",
        "//kglib/graph/test",
        "//kglib/graph/mock"
    ]
)

py_test(
    name = "from_queries_it",
    srcs = [
        "from_queries_it.py"
    ],
    deps = [
        "create",
        "//kglib/graph/test",
        "//kglib/graph/mock"
    ],
    data = [
        "@graknlabs_grakn_core//:assemble-mac-zip",
    ]
)


py_library(
    name = "create",
    srcs = glob(['**/*.py']),
    deps = [
        requirement('networkx'),
        requirement('decorator'),
        '//kglib/kgcn',
        '//kglib/graph/create/model',
    ],
    visibility=['//visibility:public']
)