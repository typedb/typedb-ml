load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

def graknlabs_build_tools():
    git_repository(
        name = "graknlabs_build_tools",
        remote = "https://github.com/graknlabs/build-tools",
        commit = "80a21b35f7e700ef07223632f02eb8894df7632f", # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_build_tools
    )


def graknlabs_grakn_core():
    git_repository(
        name="graknlabs_grakn_core",
        remote="https://github.com/graknlabs/grakn",
        commit="46f7b209e005349fd46f62aec1b95163ab47a486"  # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_grakn_core
    )

def graknlabs_client_python():
    git_repository(
        name = "graknlabs_client_python",
        remote = "https://github.com/graknlabs/client-python",
        commit = "963dd17110a5f22a567fa9ab3858271be33a6618" # sync-marker: do not remove this comment, this is used for sync-dependencies by @graknlabs_client_python
    )
