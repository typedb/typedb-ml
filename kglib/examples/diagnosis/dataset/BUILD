load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl", vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_library(
    name = "dataset",
    srcs = [
        'generate.py',
        'pmf.py',
    ],
    deps = [
        "@vaticle_typedb_client_python//:client_python",
        vaticle_kglib_requirement('numpy'),
        vaticle_kglib_requirement('pandas'),
        vaticle_kglib_requirement('pytz'),
        vaticle_kglib_requirement('python-dateutil'),
        vaticle_kglib_requirement('six')
    ],
    visibility=['//visibility:public']
)

filegroup(
    name = "diagnosis-example-typeql-files",
    srcs = [
        "schema.tql",
        "seed_data.tql",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "pmf_test",
    srcs = [
        "pmf_test.py"
    ],
    deps = [
        "dataset"
    ]
)

checkstyle_test(
    name = "checkstyle",
    include = glob([
        "*",
        "diagnosis/*"
    ]),
    license_type = "apache-header",
)
