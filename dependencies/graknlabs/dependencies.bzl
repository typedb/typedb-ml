load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def graknlabs_build_tools():
    git_repository(
        name = "graknlabs_build_tools",
        remote = "https://github.com/graknlabs/build-tools",
        commit = "54109ff7428d0b7c04bd30ecff6f7e77f32c1bfd", # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_build_tools
    )


def graknlabs_grakn_core():
    git_repository(
        name = "graknlabs_grakn_core",
        remote = "https://github.com/graknlabs/grakn",
        commit = "bda899eb586106c1d13a4e44afa06c4966941c67"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_grakn_core
    )

def graknlabs_client_python():
    git_repository(
        name = "graknlabs_client_python",
        remote = "https://github.com/graknlabs/client-python",
        commit = "9a3decb7952d021e5d12547165ad49eadd409afc" # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_client_python
    )
