load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")


def io_bazel_rules_python():
    git_repository(
        name = "io_bazel_rules_python",
        # Grakn python rules
        remote = "https://github.com/graknlabs/rules_python.git",
        commit = "4443fa25feac79b0e4c7c63ca84f87a1d6032f49"  
    )
