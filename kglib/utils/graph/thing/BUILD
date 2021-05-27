load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")


py_test(
    name = "queries_to_networkx_graph_test",
    srcs = [
        "queries_to_networkx_graph_test.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/grakn/test",
        "//kglib/utils/graph/test",
    ]
)

py_test(
    name = "queries_to_networkx_graph_it",
    srcs = [
        "queries_to_networkx_graph_it.py"
    ],
    deps = [
        "thing",
        "//kglib/utils/grakn/test",
        "//kglib/utils/graph/test",
        "@vaticle_typedb_client_python//:client_python",
    ],
    data = ["@vaticle_typedb_artifact_linux//file"],
    args = ["$(location @vaticle_typedb_artifact_linux//file)"],
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
    srcs = [
        'concept_dict_to_networkx_graph.py',
        'queries_to_networkx_graph.py'
    ],
    deps = [
        "//kglib/utils/grakn/object",
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('decorator'),
    ],
    visibility=['//visibility:public']
)