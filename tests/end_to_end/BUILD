load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "diagnosis",
    size = "enormous",
    srcs = [
        "kgcn/diagnosis.py"
    ],
    deps = [
        "//kglib/kgcn/examples/diagnosis",
        "//kglib/utils/grakn/test",
    ],
    data = [
        "//kglib/utils/grakn/synthetic/examples:graql-schema",
        "@graknlabs_grakn_core_artifact_linux//file"
    ],
    args = [
        "$(locations //kglib/utils/grakn/synthetic/examples:graql-schema)",
        "$(location @graknlabs_grakn_core_artifact_linux//file)",
    ],
)
