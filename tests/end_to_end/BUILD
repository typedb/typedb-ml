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
        "@graknlabs_grakn_core//:assemble-mac-zip",
        "//kglib/utils/grakn/synthetic/examples:diagnosis/schema.gql"
    ]
)
