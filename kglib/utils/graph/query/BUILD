load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")


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
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)