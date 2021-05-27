load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

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