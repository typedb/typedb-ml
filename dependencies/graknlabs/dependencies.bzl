load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def graknlabs_build_tools():
    git_repository(
        name = "graknlabs_build_tools",
        remote = "https://github.com/graknlabs/build-tools",
        commit = "c71c07b9ff704f2521319b77d4dbfa27606182f9", # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_build_tools
    )


def graknlabs_grakn_core():
    git_repository(
        name = "graknlabs_grakn_core",
        remote = "https://github.com/graknlabs/grakn",
        commit = "37aaeea29f4d6dfe4b5dddb3917835c300442802"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_grakn_core
    )

def graknlabs_client_python():
    git_repository(
        name = "graknlabs_client_python",
        remote = "https://github.com/graknlabs/client-python",
        commit = "a006034518564240f9c8c4fe1455855c29678534" # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_client_python
    )
