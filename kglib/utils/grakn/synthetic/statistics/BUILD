load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

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
        graknlabs_kglib_requirement('numpy'),
        graknlabs_kglib_requirement('pandas'),
        graknlabs_kglib_requirement('pytz'),
        graknlabs_kglib_requirement('python-dateutil'),
        graknlabs_kglib_requirement('six')
    ],
    visibility=['//visibility:public']
)