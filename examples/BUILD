load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@pypi_dependencies//:requirements.bzl", "requirement")

py_test(
    name = "test_pypi_end_to_end_test",
    main = "end_to_end_test.py",
    srcs = [
        "kgcn/animal_trade/test/end_to_end_test.py"
    ],
    deps = [
        "test-pypi-kglib",
    ],
    data = [
        "@animaltrade_dist//file",
    ]
)

py_library(
    name = "test-pypi-kglib",
    srcs = [
        "kgcn/animal_trade/test/end_to_end_test.py"
    ],
    deps = [
        requirement('grakn-kglib'),

        # Grakn deps
        requirement('grakn'),
        requirement('grpcio'),

        # TensorFlow deps
        requirement('tensorflow'),
        requirement('numpy'),
        requirement('protobuf'),
        requirement('six'),
        requirement('absl-py'),
        requirement('keras_applications'),
        requirement('keras_preprocessing'),
        requirement('gast'),
        requirement('astor'),
        requirement('termcolor'),

        requirement('tensorflow-hub'),
        requirement('scikit-learn'),
        requirement('scipy'),
    ]
)