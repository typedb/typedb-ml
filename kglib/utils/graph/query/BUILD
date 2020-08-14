load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")


py_test(
    name = "query_graph_test",
    srcs = [
        "query_graph_test.py"
    ],
    deps = [
        "query",
    ],
)

py_library(
    name = "query",
    srcs = [
        'query_graph.py',
    ],
    deps = [
        graknlabs_kglib_requirement('networkx'),
        graknlabs_kglib_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)