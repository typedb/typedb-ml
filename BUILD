exports_files(["requirements.txt"])

load("@io_bazel_rules_python//python:python.bzl", "py_library", "py_test")
load("@pypi_dependencies//:requirements.bzl", "requirement")
load("@graknlabs_bazel_distribution_pip//:requirements.bzl", deployment_requirement = "requirement")

load("@graknlabs_bazel_distribution//pip:rules.bzl", "assemble_pip", "deploy_pip")

assemble_pip(
    name = "assemble-pip",
    target = "//kglib:kglib",
    version_file = "//:VERSION",
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
    install_requires=['absl-py==0.5.0', 'astor==0.7.1', 'cycler==0.10.0', 'decorator==4.3.0', 'gast==0.2.0',
                      'grakn-client==1.5.3',
                      'graph-nets==1.0.4', 'h5py==2.8.0', 'Keras-Applications==1.0.6', 'Keras-Preprocessing==1.0.5',
                      'Markdown==3.0.1', 'matplotlib==3.1.0', 'networkx==2.2', 'numpy==1.16.4', 'pandas==0.24.1',
                      'protobuf==3.6.1', 'python-dateutil==2.8.0', 'pyparsing==2.4.0', 'pytz==2018.9',
                      'scikit-learn==0.20.1', 'semantic-version==2.6.0',
                      'scipy==1.1.0', 'six==1.11.0', 'tensorboard==1.13.1', 'tensorflow==1.13.1',
                      'tensorflow-probability==0.6.0',
                      'tensorflow-hub==0.1.1', 'termcolor==1.1.0', 'Werkzeug==0.14.1', 'wrapt==1.11.1',
                      'grpcio==1.16.0', 'protobuf==3.6.1', 'six==1.11.0', 'enum34==1.1.6', 'twine==1.12.1', 'requests==2.21.0'],
    keywords = ["machine learning", "logical reasoning", "knowledege graph", "grakn", "database", "graph",
                "knowledgebase", "knowledge-engineering"],

    description = "A Machine Learning Library for the Grakn knowledge graph.",
    long_description_file = "//:README.md",
)

deploy_pip(
    name = "deploy-pip",
    target = ":assemble-pip",
    deployment_properties = "@graknlabs_build_tools//:deployment.properties",
)
