
workspace(
    name = "kglib"
)


########################################################################################################################
# Load Bazel Rules
########################################################################################################################

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

# We require loading bazel_rules from the graknlabs fork, not from bazel, since we've patched the python rules to work with TensorFlow
load("//dependencies/graknlabs:dependencies.bzl", "io_bazel_rules_python")
io_bazel_rules_python()

## Only needed for PIP support:
load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip3_import")
pip_repositories()


########################################################################################################################
# Load Bazel Distribution
########################################################################################################################

# Load Bazel Distribution here, since it is required for kglib and for grakn
git_repository(
    name="graknlabs_bazel_distribution",
    remote="https://github.com/graknlabs/bazel-distribution",
    commit="62a9a6343e9f2a1aeed7c935e9092c0fd1e8e8ac"
)

# --- Load the dependencies of graknlabs_bazel_distribution ---

load("@graknlabs_bazel_distribution//github:dependencies.bzl", "github_dependencies_for_deployment")
github_dependencies_for_deployment()

#load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "io_bazel_skydoc",
    remote = "https://github.com/graknlabs/skydoc.git",
    branch = "experimental-skydoc-allow-dep-on-bazel-tools",
)

load("@io_bazel_skydoc//:setup.bzl", "skydoc_repositories")
skydoc_repositories()

load("@io_bazel_rules_sass//:package.bzl", "rules_sass_dependencies")
rules_sass_dependencies()

load("@build_bazel_rules_nodejs//:defs.bzl", "node_repositories")
node_repositories()

load("@io_bazel_rules_sass//:defs.bzl", "sass_repositories")
sass_repositories()

# Skip these graknlabs_bazel_distribution dependencies since they are already present
#git_repository(
#    name = "io_bazel_rules_python",
#    remote = "https://github.com/bazelbuild/rules_python.git",
#    commit = "fdbb17a4118a1728d19e638a5291b4c4266ea5b8",
#)

#load("@io_bazel_rules_python//python:pip.bzl", "pip_repositories", "pip3_import")
#pip_repositories()

pip3_import(
    name = "graknlabs_bazel_distribution_pip",
    requirements = "@graknlabs_bazel_distribution//pip:requirements.txt",
)
load("@graknlabs_bazel_distribution_pip//:requirements.bzl", graknlabs_bazel_distribution_pip_install = "pip_install")
graknlabs_bazel_distribution_pip_install()


########################################################################################################################
# Load KGLIB's PyPi requirements
########################################################################################################################

# Load PyPI dependencies for Python programs
pip3_import(
    name = "pypi_dependencies",
    requirements = "//:requirements.txt",
)

load("@pypi_dependencies//:requirements.bzl", pip_install_kglib_requirements = "pip_install")
pip_install_kglib_requirements()


########################################################################################################################
# Load the pre-loaded Animal Trade Grakn distribution
########################################################################################################################

http_file(
  name = "animaltrade_dist",
  urls = ["https://storage.googleapis.com/kglib/grakn-core-all-mac-animaltrade1.5.3.zip", # TODO How to update to the latest relase each time?
  ]
)


########################################################################################################################
# Load Grakn
########################################################################################################################

git_repository(
    name="graknlabs_grakn_core",
    remote="https://github.com/graknlabs/grakn",
    commit="2845bb009876a74896bd479a7e49955c7fa1c7ca"
)


################################
# Load Grakn Labs dependencies #
################################

load(
    "@graknlabs_grakn_core//dependencies/graknlabs:dependencies.bzl",
    "graknlabs_build_tools",
    "graknlabs_graql",
    "graknlabs_protocol",
    "graknlabs_client_java",
    "graknlabs_benchmark"
)
graknlabs_build_tools()
graknlabs_graql()
graknlabs_protocol()
graknlabs_client_java()
graknlabs_benchmark()

# Skip since these are already present above
#load("@graknlabs_build_tools//distribution:dependencies.bzl", "graknlabs_bazel_distribution")
#graknlabs_bazel_distribution()

###########################
# Load Bazel dependencies #
###########################

load("@graknlabs_build_tools//bazel:dependencies.bzl", "bazel_common", "bazel_deps", "bazel_toolchain")
bazel_common()
bazel_deps()
bazel_toolchain()


#################################
# Load Build Tools dependencies #
#################################

load("@graknlabs_build_tools//checkstyle:dependencies.bzl", "checkstyle_dependencies")
checkstyle_dependencies()

load("@graknlabs_build_tools//sonarcloud:dependencies.bzl", "sonarcloud_dependencies")
sonarcloud_dependencies()

load("@graknlabs_build_tools//bazel:dependencies.bzl", "bazel_rules_python")
bazel_rules_python()

pip3_import(
    name = "graknlabs_build_tools_ci_pip",
    requirements = "@graknlabs_build_tools//ci:requirements.txt",
)

load("@graknlabs_build_tools_ci_pip//:requirements.bzl",
graknlabs_build_tools_ci_pip_install = "pip_install")
graknlabs_build_tools_ci_pip_install()


#####################################
# Load Java dependencies from Maven #
#####################################

load("@graknlabs_grakn_core//dependencies/maven:dependencies.bzl", "maven_dependencies")
maven_dependencies()


###########################
# Load Graql dependencies #
###########################

# Load ANTLR dependencies for Bazel
load("@graknlabs_graql//dependencies/compilers:dependencies.bzl", "antlr_dependencies")
antlr_dependencies()

# Load ANTLR dependencies for ANTLR programs
load("@rules_antlr//antlr:deps.bzl", "antlr_dependencies")
antlr_dependencies()

load("@graknlabs_graql//dependencies/maven:dependencies.bzl",
graknlabs_graql_maven_dependencies = "maven_dependencies")
graknlabs_graql_maven_dependencies()


###########################
# Load Benchmark dependencies #
###########################
load("@graknlabs_benchmark//dependencies/maven:dependencies.bzl",
graknlabs_benchmark_maven_dependencies = "maven_dependencies")
graknlabs_benchmark_maven_dependencies()


#######################################
# Load compiler dependencies for GRPC #
#######################################

load("@graknlabs_build_tools//grpc:dependencies.bzl", "grpc_dependencies")
grpc_dependencies()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl",
com_github_grpc_grpc_deps = "grpc_deps")
com_github_grpc_grpc_deps()

load("@stackb_rules_proto//java:deps.bzl", "java_grpc_compile")
java_grpc_compile()


##################################
# Load Distribution dependencies #
##################################

# Skip since these are already present above
# TODO: rename the macro we load here to deploy_github_dependencies
#load("@graknlabs_bazel_distribution//github:dependencies.bzl", "github_dependencies_for_deployment")
#github_dependencies_for_deployment()

load("@graknlabs_build_tools//bazel:dependencies.bzl", "bazel_rules_docker")
bazel_rules_docker()

load("@io_bazel_rules_docker//repositories:repositories.bzl",
bazel_rules_docker_repositories = "repositories")
bazel_rules_docker_repositories()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")
container_pull(
  name = "openjdk_image",
  registry = "index.docker.io",
  repository = "library/openjdk",
  tag = "8"
)

#####################################
# Load Bazel common workspace rules #
#####################################

# TODO: Figure out why this cannot be loaded at earlier at the top of the file
load("@com_github_google_bazel_common//:workspace_defs.bzl", "google_common_workspace_rules")
google_common_workspace_rules()
