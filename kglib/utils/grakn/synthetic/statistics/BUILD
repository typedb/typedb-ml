load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_test(
    name = "pmf_test",
    srcs = [
        "pmf_test.py"
    ],
    deps = [
        "statistics"
    ]
)

py_library(
    name = "statistics",
    srcs = [
        'pmf.py',
    ],
    deps = [
        vaticle_kglib_requirement('numpy'),
        vaticle_kglib_requirement('pandas'),
        vaticle_kglib_requirement('pytz'),
        vaticle_kglib_requirement('python-dateutil'),
        vaticle_kglib_requirement('six')
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
