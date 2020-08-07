load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

py_test(
    name = "metrics_test",
    srcs = [
        "metrics_test.py"
    ],
    deps = [
        "learn"
    ]
)

py_test(
    name = "learn_IT",
    srcs = [
        "learn_IT.py"
    ],
    deps = [
        "learn",
        "//kglib/kgcn/models",
        "//kglib/kgcn/plot"
    ]
)

py_library(
    name = "learn",
    srcs = [
        'feed.py',
        'learn.py',
        'loss.py',
        'metrics.py',
    ],
    deps = [
        # Networkx deps
        graknlabs_kglib_requirement('networkx'),
        graknlabs_kglib_requirement('decorator'),

        # Graph nets deps
        graknlabs_kglib_requirement('absl-py'),
        graknlabs_kglib_requirement('cloudpickle'),
        graknlabs_kglib_requirement('contextlib2'),
        # graknlabs_kglib_requirement('decorator'),
        graknlabs_kglib_requirement('dm-sonnet'),
        graknlabs_kglib_requirement('future'),
        graknlabs_kglib_requirement('graph-nets'),
        # graknlabs_kglib_requirement('networkx'),
        graknlabs_kglib_requirement('numpy'),
        graknlabs_kglib_requirement('semantic-version'),
        graknlabs_kglib_requirement('six'),
        graknlabs_kglib_requirement('tensorflow-probability'),
        graknlabs_kglib_requirement('wrapt'),

        # Tensorflow deps
        # graknlabs_kglib_requirement('absl-py'),
        graknlabs_kglib_requirement('astor'),
        graknlabs_kglib_requirement('gast'),
        graknlabs_kglib_requirement('google-pasta'),
        graknlabs_kglib_requirement('keras-applications'),
        graknlabs_kglib_requirement('keras-preprocessing'),
        # graknlabs_kglib_requirement('six'),
        graknlabs_kglib_requirement('protobuf'),
        graknlabs_kglib_requirement('tensorboard'),
        graknlabs_kglib_requirement('tensorflow'),
        graknlabs_kglib_requirement('tensorflow-estimator'),
        graknlabs_kglib_requirement('termcolor'),
        # graknlabs_kglib_requirement('wrapt'),

        # Scipy deps
        graknlabs_kglib_requirement('scipy')

    ],
    visibility=['//visibility:public']
)