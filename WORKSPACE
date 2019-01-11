workspace(
    name = "research"
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

#git_repository(
#    name = "io_bazel_rules_python",
#    # Bazel python rules
##    remote = "https://github.com/bazelbuild/rules_python.git",
##    commit = "cc4cbf2f042695f4d1d4198c22459b3dbe7f8e43",
#
#    # Grakn python rules
#    remote = "https://github.com/graknlabs/rules_python.git",
#    commit = "abd475a72ae6a098cc9f859eb435dddd992bc884",
#)

# Use a patched local version of rules_python
local_repository(
    name = "io_bazel_rules_python",
    path = "/Users/jamesfletcher/programming/rules_python"
)

## Only needed for PIP support:
load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip3_import")
pip_repositories()

# Load PyPI dependencies for Python programs
pip3_import(
    name = "pypi_dependencies",
#    requirements = "//dependencies/pip:requirements.txt",
    requirements = "//:requirements.txt",
)
load("@pypi_dependencies//:requirements.bzl", "pip_install")
pip_install()