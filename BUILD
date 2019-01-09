

py_test(
    name = "my_test",
    srcs = [
        "kgcn/my_test.py"
    ],
    deps = [
    ]
)

py_test(
    name = "ordered_test",
    srcs = [
        "kgcn/neighbourhood/data/sampling/ordered_test.py"
    ],
    deps = [
        "kgcn"
    ]
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
    ]
)

py_test(
    name = "schema_test",
    srcs = [
        "kgcn/encoder/schema_test.py"
    ],
    deps = [
        "kgcn",
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
    srcs = glob(['kgcn/**/*.py'])
)