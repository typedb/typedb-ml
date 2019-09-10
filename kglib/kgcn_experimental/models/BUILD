load("@io_bazel_rules_python//python:python.bzl", "py_test", "py_library")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "attribute_test",
    srcs = [
        "attribute_test.py"
    ],
    deps = [
        "models"
    ]
)

py_library(
    name = "models",
    srcs = glob(['**/*.py']),
    deps = [
        '//kglib/utils/test',
#        requirement('networkx'),
#        requirement('decorator'),
#        requirement('numpy'),
#        requirement('matplotlib'),
#        requirement('pyparsing'),
#        requirement('cycler'),
#        requirement('python-dateutil'),
#        requirement('kiwisolver'),
#        requirement('graph_nets'),
        requirement('dm-sonnet'),
        requirement('tensorflow'),
        requirement('tensorflow-probability'),
        requirement('semantic-version'),
        requirement('contextlib2'),
        requirement('wrapt')

    ],
    visibility=['//visibility:public']
)