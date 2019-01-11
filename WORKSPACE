workspace(
    name = "kgcn"
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "io_bazel_rules_python",
    # Grakn python rules
    remote = "https://github.com/graknlabs/rules_python.git",
    commit = "4443fa25feac79b0e4c7c63ca84f87a1d6032f49",
)

git_repository(
    name="graknlabs_rules_deployment",
    remote="https://github.com/graknlabs/deployment",
    commit="8d68b4f13fe063ed7ccd04c29ab5f91e81fba052"
)

## Only needed for PIP support:
load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip3_import")
pip_repositories()

# Load PyPI dependencies for Python programs
pip3_import(
    name = "pypi_dependencies",
    requirements = "//:requirements.txt",
)
load("@pypi_dependencies//:requirements.bzl", "pip_install")
pip_install()