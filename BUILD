exports_files(["requirements.txt", "RELEASE_TEMPLATE.md"])

load("@rules_python//python:defs.bzl", "py_library", "py_test")

load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

load("@vaticle_bazel_distribution//github:rules.bzl", "deploy_github")
load("@vaticle_bazel_distribution//pip:rules.bzl", "assemble_pip", "deploy_pip")
load("@vaticle_kglib_pip//:requirements.bzl",
       vaticle_kglib_requirement = "requirement")

load("@vaticle_dependencies//distribution:deployment.bzl", "deployment")
load("//:deployment.bzl", github_deployment = "deployment")
load("@vaticle_dependencies//tool/release:rules.bzl", "release_validate_deps")


assemble_pip(
    name = "assemble-pip",
    target = "//kglib:kglib",
    package_name = "vaticle-kglib",
    classifiers = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    url = "https://github.com/vaticle/kglib",
    author = "Vaticle",
    author_email = "community@vaticle.com",
    license = "Apache-2.0",
    requirements_file = "//:requirements.txt",
    keywords = ["machine learning", "logical reasoning", "knowledege graph", "grakn", "database", "graph",
                "knowledgebase", "knowledge-engineering"],

    description = "A Machine Learning Library for TypeDB.",
    long_description_file = "//:README.md",
)

deploy_pip(
    name = "deploy-pip",
    target = ":assemble-pip",
    snapshot = deployment["pypi.snapshot"],
    release = deployment["pypi.release"],
)

release_validate_deps(
    name = "release-validate-deps",
    refs = "@vaticle_kglib_workspace_refs//:refs.json",
    tagged_deps = [
        "@vaticle_typedb",
        "@vaticle_typedb_client_python",
    ],
    tags = ["manual"]
)
