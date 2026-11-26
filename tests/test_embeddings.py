"""Tests for the model-agnostic embedding pipeline.

A real encoder (DINOv2, BiomedCLIP) needs PyTorch and a large model download,
so these tests use a tiny fake encoder instead: it "embeds" each image as its
average RGB colour. That is enough to check that images are loaded, batched,
kept in order and saved correctly, without any deep-learning dependency.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from skin_lesion_triage.embeddings import (
    compute_embeddings,
    load_embeddings,
    save_embeddings,
)

# One solid colour per fake image, so each image has a known "embedding".
COLOURS = {
    "ISIC_0000000": (255, 0, 0),
    "ISIC_0000001": (0, 255, 0),
    "ISIC_0000002": (0, 0, 255),
    "ISIC_0000003": (10, 20, 30),
    "ISIC_0000004": (200, 100, 50),
}


class FakeEncoder:
    """Embeds each image as its mean (R, G, B) and records the batch sizes it saw."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        self.batch_sizes.append(len(images))
        return np.stack([np.asarray(img, dtype=np.float32).mean(axis=(0, 1)) for img in images])


@pytest.fixture
def images_dir(tmp_path: Path) -> Path:
    """A folder with one small solid-colour JPEG per entry in COLOURS."""
    for image_id, colour in COLOURS.items():
        Image.new("RGB", (8, 8), colour).save(tmp_path / f"{image_id}.jpg")
    return tmp_path


def test_one_row_per_image_in_the_requested_order(images_dir):
    # Ask in a different order than the files were created.
    image_ids = ["ISIC_0000002", "ISIC_0000000", "ISIC_0000004"]

    embeddings = compute_embeddings(image_ids, images_dir, FakeEncoder())

    assert embeddings.shape == (3, 3)
    expected = np.array([COLOURS[i] for i in image_ids], dtype=np.float32)
    # JPEG compression changes colours slightly, hence the tolerance.
    np.testing.assert_allclose(embeddings, expected, atol=3)


def test_images_are_sent_to_the_encoder_in_batches(images_dir):
    encoder = FakeEncoder()

    compute_embeddings(list(COLOURS), images_dir, encoder, batch_size=2)

    assert encoder.batch_sizes == [2, 2, 1]


def test_result_does_not_depend_on_batch_size(images_dir):
    one_by_one = compute_embeddings(list(COLOURS), images_dir, FakeEncoder(), batch_size=1)
    all_at_once = compute_embeddings(list(COLOURS), images_dir, FakeEncoder(), batch_size=100)

    np.testing.assert_array_equal(one_by_one, all_at_once)


def test_images_are_converted_to_rgb(images_dir):
    # A grayscale image must reach the encoder as RGB, like every other image.
    Image.new("L", (8, 8), 128).save(images_dir / "ISIC_9999999.jpg")
    modes = []

    def encoder(images):
        modes.extend(img.mode for img in images)
        return np.zeros((len(images), 3), dtype=np.float32)

    compute_embeddings(["ISIC_9999999", "ISIC_0000000"], images_dir, encoder)

    assert modes == ["RGB", "RGB"]


def test_missing_image_raises_with_its_id(images_dir):
    with pytest.raises(FileNotFoundError, match="ISIC_1234567"):
        compute_embeddings(["ISIC_0000000", "ISIC_1234567"], images_dir, FakeEncoder())


def test_encoder_returning_wrong_number_of_rows_raises(images_dir):
    def broken_encoder(images):
        return np.zeros((len(images) + 1, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="rows"):
        compute_embeddings(list(COLOURS), images_dir, broken_encoder)


def test_invalid_batch_size_raises(images_dir):
    with pytest.raises(ValueError, match="batch_size"):
        compute_embeddings(list(COLOURS), images_dir, FakeEncoder(), batch_size=0)


def test_save_and_load_round_trip(tmp_path):
    image_ids = ["ISIC_0000000", "ISIC_0000001"]
    embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
    path = tmp_path / "embeddings.npz"

    save_embeddings(path, image_ids, embeddings)
    loaded_ids, loaded_embeddings = load_embeddings(path)

    assert loaded_ids == image_ids
    np.testing.assert_array_equal(loaded_embeddings, embeddings)


def test_save_rejects_mismatched_ids_and_rows(tmp_path):
    with pytest.raises(ValueError):
        save_embeddings(tmp_path / "e.npz", ["only_one_id"], np.zeros((2, 3)))
