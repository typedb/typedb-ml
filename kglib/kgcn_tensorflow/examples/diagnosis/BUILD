load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_test(
    name = "diagnosis_test",
    srcs = [
        "diagnosis_test.py"
    ],
    deps = [
        "diagnosis",
        "//kglib/utils/graph/test",
        vaticle_kglib_requirement('numpy'),
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('decorator'),
    ]
)

py_library(
    name = "diagnosis",
    srcs = [
        'diagnosis.py'
    ],
    deps = [
        "//kglib/kgcn_tensorflow/pipeline",
        "//kglib/kgcn_tensorflow/learn",
        "//kglib/kgcn_tensorflow/plot",
        "//kglib/kgcn_tensorflow/models",
        "//kglib/utils/grakn/synthetic",
        "//kglib/utils/grakn/type",
        "@vaticle_typedb_client_python//:client_python",
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
