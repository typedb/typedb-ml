load("@vaticle_bazel_distribution//artifact:rules.bzl", "artifact_extractor")
load("@vaticle_dependencies//tool/checkstyle:rules.bzl", "checkstyle_test")

artifact_extractor(
    name = "typedb-extractor-linux",
    artifact = "@vaticle_typedb_artifact_linux//file",
)

checkstyle_test(
    name = "checkstyle",
    include = glob(["*"]),
    license_type = "apache",
)
