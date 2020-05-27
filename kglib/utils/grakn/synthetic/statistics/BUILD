load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

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
        requirement('numpy'),
        requirement('pandas'),
        requirement('pytz'),
        requirement('python-dateutil'),
        requirement('six')
    ],
    visibility=['//visibility:public']
)