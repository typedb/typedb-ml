load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")


py_test(
    name = "queries_to_graph_test",
    srcs = [
        "queries_to_graph_test.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/grakn/test",
        "//kglib/utils/graph/test",
    ]
)

py_test(
    name = "queries_to_graph_it",
    srcs = [
        "queries_to_graph_it.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/grakn/test",
        "//kglib/utils/graph/test",
        "@graknlabs_client_python//:client_python",
    ],
    data = [
        "@graknlabs_grakn_core//:assemble-mac-zip",
    ]
)


py_test(
    name = "concept_dict_to_graph_test",
    srcs = [
        "concept_dict_to_graph_test.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/graph/test",
    ],
)

py_library(
    name = "thing",
    srcs = glob(['**/*.py']),
    deps = [
        "//kglib/utils/grakn/object",
        requirement('networkx'),
        requirement('decorator'),
    ],
    visibility=['//visibility:public']
)