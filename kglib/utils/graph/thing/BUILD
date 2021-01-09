load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")


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
    data = ["@graknlabs_grakn_core_artifact_linux//file"],
    args = ["$(location @graknlabs_grakn_core_artifact_linux//file)"],
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
    srcs = ['concept_dict_to_graph.py',
            'queries_to_graph.py'],
    deps = [
        "//kglib/utils/grakn/object",
        graknlabs_kglib_requirement('networkx'),
        graknlabs_kglib_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)