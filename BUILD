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
        "kgcn/src/neighbourhood/data/sampling/ordered_test.py"
    ],
    deps = [
        "kgcn"
    ]
)

py_test(
    name = "random_test",
    srcs = [
        "kgcn/src/neighbourhood/data/sampling/random_test.py"
    ],
    deps = [
        "kgcn",
    ]
)

py_library(
    name = "kgcn",
    srcs = glob(['**/*.py'])

)