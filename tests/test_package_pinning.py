"""Pinning a package: a full commit id for Hugging Face downloads and the digest of SHA256SUMS.json."""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from bielik_one import launch
from bielik_one.api import vllm_middleware as middleware

from tests.test_launch import dry, package

COMMIT = "567d29b0f15df28794acf0f6187f9216860d4551"


def digest(root: Path) -> str:
    return hashlib.sha256((root / "SHA256SUMS.json").read_bytes()).hexdigest()


def republish(root: Path) -> None:
    """Change a runtime file and rewrite the manifest to match, as a republished package would."""
    (root / "calibration.json").write_text('{"temperatures": "changed"}')
    sums = json.loads((root / "SHA256SUMS.json").read_text())
    sums["calibration.json"] = hashlib.sha256((root / "calibration.json").read_bytes()).hexdigest()
    (root / "SHA256SUMS.json").write_text(json.dumps(sums))


@pytest.fixture
def local_package():
    with tempfile.TemporaryDirectory() as d:
        package("qwen-merged", Path(d))
        yield Path(d)


@pytest.fixture
def hub_snapshot():
    """A package laid out like the Hugging Face cache: .../snapshots/<commit>/."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "models--speakleash--Bielik-One-4B-r1" / "snapshots" / COMMIT
        root.mkdir(parents=True)
        package("qwen-merged", root)
        yield root


def test_dry_run_prints_what_to_pin(local_package, hub_snapshot):
    out = dry("backend", str(local_package))
    assert out["manifest_sha256"] == digest(local_package)
    assert out["revision"] is None and out["manifest_pinned"] is False
    assert dry("backend", str(hub_snapshot))["revision"] == COMMIT


def test_a_republished_package_passes_the_manifest_but_not_the_pinned_digest(local_package):
    pinned = digest(local_package)
    republish(local_package)
    # The gap: files and manifest changed together still verify.
    assert dry("backend", str(local_package))["checksums_verified"] is True
    for component in ("backend", "frontend"):
        with pytest.raises(ValueError, match="this is not the package you pinned"):
            dry(component, str(local_package), "--manifest-sha256", pinned)


def test_the_pinned_digest_accepts_the_same_package(local_package):
    for value in (digest(local_package), digest(local_package).upper()):
        out = dry("frontend", str(local_package), "--manifest-sha256", value)
        assert out["manifest_pinned"] is True and out["manifest_sha256"] == digest(local_package)


@pytest.mark.parametrize("value", ["abc", "g" * 64, "0" * 63, "0" * 65])
def test_a_malformed_digest_is_refused(local_package, value):
    with pytest.raises(ValueError, match="64 hexadecimal characters"):
        dry("backend", str(local_package), "--manifest-sha256", value)


@pytest.mark.parametrize("revision", [None, "main", "v1.0", "567d29b0f1", COMMIT.upper(), COMMIT + "0"])
def test_require_pinned_revision_refuses_anything_but_a_full_commit_before_downloading(revision):
    args = ["--require-pinned-revision"] + (["--revision", revision] if revision else [])
    with mock.patch("huggingface_hub.snapshot_download") as download:
        with pytest.raises(ValueError, match="full 40-character commit id"):
            dry("backend", "speakleash/Bielik-One-4B-r1", *args)
        download.assert_not_called()


def test_require_pinned_revision_accepts_a_full_commit(hub_snapshot):
    with mock.patch("huggingface_hub.snapshot_download", return_value=str(hub_snapshot)) as download:
        out = dry("backend", "speakleash/Bielik-One-4B-r1", "--revision", COMMIT, "--require-pinned-revision")
    download.assert_called_once_with("speakleash/Bielik-One-4B-r1", revision=COMMIT)
    assert out["revision"] == COMMIT


def test_require_pinned_revision_on_a_local_directory_needs_the_digest(local_package):
    with pytest.raises(ValueError, match="local directory has no revision; pass --manifest-sha256"):
        dry("backend", str(local_package), "--require-pinned-revision")
    out = dry("backend", str(local_package), "--require-pinned-revision", "--manifest-sha256", digest(local_package))
    assert out["manifest_pinned"] is True


def test_native_checkpoints_have_no_manifest_to_pin(local_package):
    with pytest.raises(ValueError, match="does not apply to --native-tokens"):
        dry("backend", str(local_package), "--native-tokens", "--manifest-sha256", "ab" * 32)
    with pytest.raises(ValueError, match="local native checkpoint has no revision or manifest"):
        dry("backend", str(local_package), "--native-tokens", "--require-pinned-revision")


def test_without_flags_nothing_changes(local_package):
    republish(local_package)
    assert dry("backend", str(local_package))["kind"] == "qwen-merged"


# -- vLLM middleware ------------------------------------------------------------------------------------------------


def test_middleware_reads_the_pinned_digest():
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package"})
    assert settings.manifest_sha256 is None
    settings = middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MANIFEST_SHA256": "AB" * 32})
    assert settings.manifest_sha256 == "ab" * 32
    with pytest.raises(RuntimeError, match="SYSTEMONE_MANIFEST_SHA256"):
        middleware.Settings.from_env({"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MANIFEST_SHA256": "nope"})
    with pytest.raises(RuntimeError, match="does not apply to native mode"):
        middleware.Settings.from_env(
            {"SYSTEMONE_PACKAGE": "/tmp/package", "SYSTEMONE_MODE": "native", "SYSTEMONE_MANIFEST_SHA256": "ab" * 32}
        )


def test_middleware_refuses_a_package_that_does_not_match_before_anything_else():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        package("qwen-adapter", root)
        pinned = digest(root)
        republish(root)
        settings = middleware.Settings(package=root, manifest_sha256=pinned)
        with mock.patch("huggingface_hub.snapshot_download") as download, pytest.raises(
            ValueError, match="this is not the package you pinned"
        ):
            middleware.build_engine(settings)
        download.assert_not_called()


def test_files_are_checked_against_the_pinned_bytes_of_the_manifest(local_package):
    # If the package changes between the digest check and verify(), verify() still uses the manifest that was pinned.
    pinned = digest(local_package)
    _, manifest = launch.check_manifest(local_package, pinned)
    republish(local_package)  # calibration.json and SHA256SUMS.json both change on disk
    with pytest.raises(ValueError, match="Checksum mismatch: calibration.json"):
        launch.verify(local_package, manifest)
    launch.verify(local_package)  # reading the new manifest instead would accept it
