load("@rules_python//python:defs.bzl", "py_test")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_test(
    name = "diagnosis",
    size = "enormous",
    srcs = [
        "kgcn/diagnosis.py"
    ],
    deps = [
        "//kglib/kgcn_tensorflow/examples/diagnosis",
        "//kglib/utils/grakn/test",
    ],
    data = [
        "//kglib/utils/grakn/synthetic/examples:graql-schema",
        "@vaticle_typedb_artifact_linux//file"
    ],
    args = [
        "$(locations //kglib/utils/grakn/synthetic/examples:graql-schema)",
        "$(location @vaticle_typedb_artifact_linux//file)",
    ],
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
