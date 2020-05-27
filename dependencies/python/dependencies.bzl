load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def rules_python():
    git_repository(
        name = "rules_python",
        # Grakn python rules
        remote = "https://github.com/graknlabs/rules_python.git",
        commit = "ee519e17ed5265bdd2431937bd271e3b76ad5b0a"
    )
