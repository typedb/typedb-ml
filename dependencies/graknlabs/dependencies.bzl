load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def io_bazel_rules_python():
    git_repository(
        name = "io_bazel_rules_python",
        # Grakn python rules
        remote = "https://github.com/graknlabs/rules_python.git",
        commit = "4443fa25feac79b0e4c7c63ca84f87a1d6032f49"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @io_bazel_rules_python
    )

def graknlabs_bazel_distribution():
    git_repository(
        name="graknlabs_bazel_distribution",
        remote="https://github.com/graknlabs/bazel-distribution",
        commit="62a9a6343e9f2a1aeed7c935e9092c0fd1e8e8ac"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_bazel_distribution
    )

def graknlabs_grakn_core():
    git_repository(
        name="graknlabs_grakn_core",
        remote="https://github.com/graknlabs/grakn",
        tag="1.5.7"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_grakn_core
    )

def graknlabs_client_python():
    git_repository(
        name = "graknlabs_client_python",
        remote = "https://github.com/graknlabs/client-python",
        tag="1.5.3" # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_client_python
    )