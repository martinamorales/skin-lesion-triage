"""Tests for the dataset download helpers.

These tests never touch the network: the download is replaced by a fake
in-memory response using pytest's `monkeypatch` fixture.
"""

import hashlib
import io
import zipfile

import pytest

from skin_lesion_triage import download
from skin_lesion_triage.download import RemoteFile, download_file, extract_images, md5sum


def test_md5sum_matches_hashlib(tmp_path):
    path = tmp_path / "file.bin"
    path.write_bytes(b"hello")
    assert md5sum(path) == hashlib.md5(b"hello").hexdigest()


class FakeResponse(io.BytesIO):
    """Mimics the object returned by urllib.request.urlopen."""

    def __init__(self, content: bytes):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}


def fake_urlopen(content: bytes):
    return lambda request: FakeResponse(content)


def test_download_file_sends_custom_user_agent(tmp_path, monkeypatch):
    # Regression test: Dataverse returns 403 for urllib's default User-Agent.
    content = b"some data"
    remote = RemoteFile(1, "data.csv", hashlib.md5(content).hexdigest())
    seen = {}

    def capture(request):
        seen["user_agent"] = request.get_header("User-agent")
        return FakeResponse(content)

    monkeypatch.setattr(download.urllib.request, "urlopen", capture)
    download_file(remote, tmp_path)

    assert seen["user_agent"] == download.USER_AGENT


def test_download_file_verifies_checksum(tmp_path, monkeypatch):
    content = b"some data"
    remote = RemoteFile(1, "data.csv", hashlib.md5(content).hexdigest())
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(content))

    path = download_file(remote, tmp_path)

    assert path.read_bytes() == content


def test_download_file_rejects_corrupted_file(tmp_path, monkeypatch):
    remote = RemoteFile(1, "data.csv", "0" * 32)  # checksum that will never match
    monkeypatch.setattr(download.urllib.request, "urlopen", fake_urlopen(b"corrupted"))

    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file(remote, tmp_path)

    # Neither the final file nor the temporary one should be left behind.
    assert list(tmp_path.iterdir()) == []


def test_download_file_skips_existing_valid_file(tmp_path, monkeypatch):
    content = b"already here"
    (tmp_path / "data.csv").write_bytes(content)
    remote = RemoteFile(1, "data.csv", hashlib.md5(content).hexdigest())

    def fail(request):
        raise AssertionError("should not download")

    monkeypatch.setattr(download.urllib.request, "urlopen", fail)

    download_file(remote, tmp_path)  # must not raise


def test_extract_images_flattens_and_filters(tmp_path):
    zip_path = tmp_path / "images.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("folder/ISIC_0000001.jpg", b"img1")
        zf.writestr("ISIC_0000002.jpg", b"img2")
        zf.writestr("readme.txt", b"not an image")
        zf.writestr("../../evil.jpg", b"zip slip attempt")

    images_dir = tmp_path / "images"
    n = extract_images(zip_path, images_dir)

    assert n == 3
    assert sorted(p.name for p in images_dir.iterdir()) == [
        "ISIC_0000001.jpg",
        "ISIC_0000002.jpg",
        "evil.jpg",  # extracted safely *inside* images_dir
    ]
    assert not (tmp_path.parent / "evil.jpg").exists()


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def test_interrupted_extraction_leaves_no_truncated_image(tmp_path, monkeypatch):
    zip_path = make_zip(tmp_path / "images.zip", {"ISIC_0000001.jpg": b"full image"})
    images_dir = tmp_path / "images"

    def crash_mid_copy(src, dst):
        dst.write(b"half")  # write part of the file...
        raise KeyboardInterrupt  # ...then get interrupted

    monkeypatch.setattr(download.shutil, "copyfileobj", crash_mid_copy)
    with pytest.raises(KeyboardInterrupt):
        extract_images(zip_path, images_dir)

    # The final name must not exist, otherwise the next run would skip it.
    assert not (images_dir / "ISIC_0000001.jpg").exists()


def test_extraction_recovers_after_interruption(tmp_path):
    zip_path = make_zip(tmp_path / "images.zip", {"ISIC_0000001.jpg": b"full image"})
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    (images_dir / "ISIC_0000001.jpg.part").write_bytes(b"half")  # leftover from a crash

    n = extract_images(zip_path, images_dir)

    assert n == 1
    assert (images_dir / "ISIC_0000001.jpg").read_bytes() == b"full image"
    assert not (images_dir / "ISIC_0000001.jpg.part").exists()
