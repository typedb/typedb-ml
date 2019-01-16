exports_files(["requirements.txt"])

load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@pypi_dependencies//:requirements.bzl", "requirement")
#load("@graknlabs_rules_deployment//pip:rules.bzl", "deploy_pip")

py_test(
  name = "my_test",
  srcs = [
      "kgcn/my_test.py"
  ],
  deps = [
      requirement('tensorflow'),
      requirement('numpy'),
      requirement('protobuf'),
      requirement('six'),
      requirement('absl-py'),
      requirement('keras_applications'),
      requirement('keras_preprocessing')
  ]
)

py_test(
    name = "ordered_test",
    srcs = [
        "kgcn/neighbourhood/data/sampling/ordered_test.py"
    ],
    deps = [
        "kgcn"
    ],
)

py_test(
    name = "random_sampling_test",
    srcs = [
        "kgcn/neighbourhood/data/sampling/random_sampling_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

#py_test(
#    name = "label_extraction_test",
#    srcs = [
#        "kgcn/use_cases/attribute_prediction/label_extraction_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

py_test(
    name = "metrics_test",
    srcs = [
        "kgcn/models/metrics_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_test(
    name = "tf_hub_test",
    srcs = [
        "kgcn/encoder/tf_hub_test.py"
    ],
    deps = [
        "kgcn",
        requirement('protobuf'),
        requirement('six'),
        requirement('absl-py'),
        requirement('keras_applications'),
        requirement('keras_preprocessing')
    ]
)

py_test(
    name = "schema_test",
    srcs = [
        "kgcn/encoder/schema_test.py"
    ],
    deps = [
        "kgcn",
        requirement('protobuf'),
        requirement('six'),
        requirement('absl-py'),
        requirement('keras_applications'),
        requirement('keras_preprocessing'),
        requirement('gast'),
        requirement('astor'),
        requirement('termcolor')
    ]
)

#py_test(
#    name = "encode_test",
#    srcs = [
#        "kgcn/encoder/encode_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

#py_test(
#    name = "data_traversal_test",
#    main = "traversal_test.py",
#    srcs = [
#        "kgcn/neighbourhood/data/traversal_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

#py_test(
#    name = "data_executor_test",
#    main = "executor_test.py",
#    srcs = [
#        "kgcn/neighbourhood/data/executor_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

#py_test(
#    name = "schema_traversal_test",
#    main = "traversal_test.py",
#    srcs = [
#        "kgcn/neighbourhood/schema/traversal_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

#py_test(
#    name = "raw_array_builder_test",
#    srcs = [
#        "kgcn/preprocess/raw_array_builder_test.py"
#    ],
#    deps = [
#        "kgcn",
#    ]
#)

py_library(
    name = "kgcn",
    srcs = glob(['kgcn/**/*.py']),
    deps = [
        requirement('numpy'),
        requirement('scikit-learn'),
        requirement('scipy'),
        requirement('tensorflow'),
        requirement('tensorflow-hub'),
    ]
)