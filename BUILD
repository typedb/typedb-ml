exports_files(["requirements.txt", "deployment.properties", "RELEASE_TEMPLATE.md"])

load("@rules_python//python:defs.bzl", "py_library", "py_test")

load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

load("@graknlabs_bazel_distribution//pip:rules.bzl", "assemble_pip", "deploy_pip")
load("@graknlabs_kglib_pip//:requirements.bzl",
       graknlabs_kglib_requirement = "requirement")

load("@graknlabs_bazel_distribution//github:rules.bzl", "deploy_github")
load("@graknlabs_dependencies//distribution/artifact:rules.bzl", "artifact_extractor")

# Python toolchain
load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")

py_runtime(
    name = "my_py2_runtime",
    interpreter_path = "/opt/circleci/.pyenv/shims/python",
    python_version = "PY2",
)

py_runtime(
    name = "my_py3_runtime",
    interpreter_path = "/opt/circleci/.pyenv/shims/python3",
    python_version = "PY3",
)

py_runtime_pair(
    name = "my_py_runtime_pair",
    py2_runtime = ":my_py2_runtime",
    py3_runtime = ":my_py3_runtime",
)

toolchain(
    name = "my_toolchain",
    toolchain = ":my_py_runtime_pair",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type",
)
# Python toolchain

assemble_pip(
    name = "assemble-pip",
    target = "//kglib:kglib",
    package_name = "grakn-kglib",
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
    url = "https://github.com/graknlabs/kglib",
    author = "Grakn Labs",
    author_email = "community@grakn.ai",
    license = "Apache-2.0",
    install_requires=[
        'enum-compat==0.0.2',
        'grakn-client==1.7.2',
        'absl-py==0.8.0',
        'astor==0.8.0',
        'cloudpickle==1.2.2',
        'contextlib2==0.5.5',
        'cycler==0.10.0',
        'decorator==4.4.0',
        'dm-sonnet==1.35',
        'future==0.17.1',
        'gast==0.3.1',
        'google-pasta==0.1.7',
        'graph-nets==1.0.4',
        'grpcio==1.23.0',
        'h5py==2.10.0',
        'Keras-Applications==1.0.8',
        'Keras-Preprocessing==1.1.0',
        'kiwisolver==1.1.0',
        'Markdown==3.1.1',
        'matplotlib==3.1.1',
        'networkx==2.3',
        'numpy==1.17.2',
        'pandas==0.25.1',
        'protobuf==3.9.1',
        'pyparsing==2.4.2',
        'python-dateutil==2.8.0',
        'pytz==2019.2',
        'scipy==1.3.1',
        'semantic-version==2.8.2',
        'six==1.12.0',
        'tensorboard==1.14.0',
        'tensorflow==1.14.0',
        'tensorflow-estimator==1.14.0',
        'tensorflow-probability==0.7.0',
        'termcolor==1.1.0',
        'Werkzeug==0.15.6',
        'wrapt==1.11.2',
    ],
    keywords = ["machine learning", "logical reasoning", "knowledege graph", "grakn", "database", "graph",
                "knowledgebase", "knowledge-engineering"],

    description = "A Machine Learning Library for the Grakn knowledge graph.",
    long_description_file = "//:README.md",
)

deploy_pip(
    name = "deploy-pip",
    target = ":assemble-pip",
    deployment_properties = "@graknlabs_dependencies//distribution:deployment.properties",
)
