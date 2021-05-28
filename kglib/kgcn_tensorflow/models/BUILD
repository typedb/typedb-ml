load("@rules_python//python:defs.bzl", "py_test", "py_library")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

py_test(
    name = "attribute_test",
    srcs = [
        "attribute_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "attribute_IT",
    srcs = [
        "attribute_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_test(
    name = "embedding_test",
    srcs = [
        "embedding_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "embedding_IT",
    srcs = [
        "embedding_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_test(
    name = "typewise_test",
    srcs = [
        "typewise_test.py"
    ],
    deps = [
        "//kglib/utils/test",
        "models"
    ]
)

py_test(
    name = "typewise_IT",
    srcs = [
        "typewise_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_test(
    name = "core_IT",
    srcs = [
        "core_IT.py"
    ],
    deps = [
        "models"
    ]
)

py_library(
    name = "models",
    srcs = [
        'attribute.py',
        'core.py',
        'embedding.py',
        'typewise.py',
    ],
    deps = [
        # Networkx deps
        vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('decorator'),

        # Graph nets deps
        vaticle_kglib_requirement('absl-py'),
        vaticle_kglib_requirement('cloudpickle'),
        vaticle_kglib_requirement('contextlib2'),
        # vaticle_kglib_requirement('decorator'),
        vaticle_kglib_requirement('dm-sonnet'),
        vaticle_kglib_requirement('future'),
        vaticle_kglib_requirement('graph-nets'),
        # vaticle_kglib_requirement('networkx'),
        vaticle_kglib_requirement('numpy'),
        vaticle_kglib_requirement('semantic-version'),
        vaticle_kglib_requirement('six'),
        vaticle_kglib_requirement('tensorflow-probability'),
        vaticle_kglib_requirement('wrapt'),

        # Tensorflow deps
        # vaticle_kglib_requirement('absl-py'),
        vaticle_kglib_requirement('astor'),
        vaticle_kglib_requirement('gast'),
        vaticle_kglib_requirement('google-pasta'),
        vaticle_kglib_requirement('keras-applications'),
        vaticle_kglib_requirement('keras-preprocessing'),
        # vaticle_kglib_requirement('six'),
        vaticle_kglib_requirement('protobuf'),
        vaticle_kglib_requirement('tensorboard'),
        vaticle_kglib_requirement('tensorflow'),
        vaticle_kglib_requirement('tensorflow-estimator'),
        vaticle_kglib_requirement('termcolor'),
        # vaticle_kglib_requirement('wrapt'),
    ],
    visibility=['//visibility:public']
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
