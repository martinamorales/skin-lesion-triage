"""Download the HAM10000 dataset from Harvard Dataverse.

Usage (from the repository root):

    uv run python -m skin_lesion_triage.download

The dataset is licensed CC BY-NC 4.0 and must not be redistributed, which is why it
is downloaded on demand instead of being committed to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# `format=original` returns the file as uploaded (e.g. CSV) instead of Dataverse's
# internal tab-separated conversion.
DATAVERSE_URL = "https://dataverse.harvard.edu/api/access/datafile/{file_id}?format=original"
# Dataverse rejects urllib's default User-Agent ("Python-urllib/x.y") with HTTP 403,
# so we identify the client explicitly.
USER_AGENT = "skin-lesion-triage/0.1 (dataset downloader)"
DEFAULT_DATA_DIR = Path("data") / "ham10000"
CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass(frozen=True)
class RemoteFile:
    """A file hosted on Dataverse, identified by its numeric id."""

    file_id: int
    filename: str
    md5: str


# File ids and checksums from doi:10.7910/DVN/DBW86T (dataset version 4.0).
HAM10000_FILES = (
    RemoteFile(4338392, "HAM10000_metadata.csv", "8f85fb1aa29d80a2797247e434deb79d"),
    RemoteFile(3172585, "HAM10000_images_part_1.zip", "4639bfa73ab251610530a97c898e6e46"),
    RemoteFile(3172584, "HAM10000_images_part_2.zip", "da43d6cc50f6613013be07e8986b384b"),
)


def md5sum(path: Path) -> str:
    """Return the MD5 hex digest of a file, reading it in chunks to save memory."""
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(remote: RemoteFile, dest_dir: Path) -> Path:
    """Download `remote` into `dest_dir` and verify its checksum.

    Skips the download if a file with the correct checksum already exists.
    Raises ValueError if the downloaded file is corrupted.
    """
    dest = dest_dir / remote.filename
    if dest.exists() and md5sum(dest) == remote.md5:
        print(f"[skip] {remote.filename} (already downloaded)")
        return dest

    # Download to a temporary ".part" file so an interrupted download never
    # leaves a half-written file with the final name.
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(
        DATAVERSE_URL.format(file_id=remote.file_id),
        headers={"User-Agent": USER_AGENT},
    )
    print(f"[download] {remote.filename}")
    with urllib.request.urlopen(request) as response, tmp.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:,.0f} MB", end="", flush=True)
    print()

    actual = md5sum(tmp)
    if actual != remote.md5:
        tmp.unlink()
        raise ValueError(
            f"Checksum mismatch for {remote.filename}: expected {remote.md5}, got {actual}"
        )
    tmp.replace(dest)
    return dest


def extract_images(zip_path: Path, images_dir: Path) -> int:
    """Extract the .jpg files of `zip_path` into a flat `images_dir`.

    Only the base file name of each member is used, which flattens any folders
    inside the archive and prevents malicious paths such as "../../evil.jpg"
    from writing outside `images_dir` ("zip slip").

    Each image is first written to a temporary ".part" file and only renamed once it
    is complete, so an interrupted extraction never leaves a truncated image that
    would be wrongly skipped on the next run.

    Returns the number of newly extracted images.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".jpg"):
                continue
            target = images_dir / name
            if target.exists():
                continue
            tmp = target.with_name(target.name + ".part")
            with zf.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            tmp.replace(target)
            extracted += 1
    return extracted


def download_ham10000(data_dir: Path = DEFAULT_DATA_DIR) -> None:
    """Download the HAM10000 metadata and images into `data_dir`."""
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    for remote in HAM10000_FILES:
        path = download_file(remote, data_dir)
        if path.suffix == ".zip":
            n = extract_images(path, images_dir)
            print(f"[extract] {n} new images from {path.name}")

    total = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"Done: {total} images in {images_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"where to store the dataset (default: {DEFAULT_DATA_DIR})",
    )
    args = parser.parse_args()
    download_ham10000(args.data_dir)


if __name__ == "__main__":
    main()
