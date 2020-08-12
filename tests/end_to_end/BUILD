load("@rules_python//python:defs.bzl", "py_test")

py_test(
    name = "diagnosis",
    size = "large",
    srcs = [
        "kgcn/diagnosis.py"
    ],
    deps = [
        "//kglib/kgcn/examples/diagnosis",
        "//kglib/utils/grakn/test",
    ],
    data = [
        "//kglib/utils/grakn/synthetic/examples:graql-schema",
        "@graknlabs_grakn_core_artifact//file"
    ],
    args = [
        "$(location @graknlabs_grakn_core_artifact//file)",
        "$(locations //kglib/utils/grakn/synthetic/examples:graql-schema)",
    ],
)
