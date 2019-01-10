workspace(
    name = "research"
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "io_bazel_rules_python",
    # Bazel python rules
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "cc4cbf2f042695f4d1d4198c22459b3dbe7f8e43",

    # Grakn python rules
#    remote = "git@github.com:graknlabs/rules_python.git",
#    commit = "abd475a72ae6a098cc9f859eb435dddd992bc884",
)

## Only needed for PIP support:
load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip_import")
pip_repositories()

# Load PyPI dependencies for Python programs
pip_import(
    name = "pypi_dependencies",
#    requirements = "//dependencies/pip:requirements.txt",
    requirements = "//:requirements.txt",
)
load("@pypi_dependencies//:requirements.bzl", "pip_install")
pip_install()