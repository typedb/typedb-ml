workspace(
    name = "kglib"
)

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "io_bazel_rules_python",
    # Grakn python rules
    remote = "https://github.com/graknlabs/rules_python.git",
    commit = "4443fa25feac79b0e4c7c63ca84f87a1d6032f49",
)

git_repository(
    name="graknlabs_bazel_distribution",
    remote="https://github.com/graknlabs/bazel-distribution",
    commit="2e932a2555d1e43f75c8ee676c926399bd12f240"
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

pip3_import(
    name = "pypi_deployment_dependencies",
    requirements = "//:deployment/requirements.txt",
)
load("@pypi_deployment_dependencies//:requirements.bzl", "pip_install")
pip_install()


load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

http_file(
  name = "animaltrade_dist",
  urls = ["https://github.com/graknlabs/kglib/releases/download/v0.1a1/grakn-animaltrade.zip", # TODO How to update to the latest relase each time?
  ]
)