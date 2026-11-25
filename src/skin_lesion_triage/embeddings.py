"""Turn HAM10000 images into embeddings with any image encoder.

An *encoder* is anything that takes a list of images and returns one row of
numbers (an embedding) per image. This module does not know which model is
inside the encoder: it only loads the images, sends them in batches, checks
the output and saves it. That keeps it free of PyTorch, so it is fast to test,
and lets us compare different foundation models (DINOv2, BiomedCLIP) by
swapping the encoder while everything else stays the same.

Embeddings are computed once (on a GPU, in Colab) and saved to a `.npz` file;
every later step (classifier, calibration, evaluation) just loads that file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image


class Encoder(Protocol):
    """The contract every encoder must follow (real or fake)."""

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        """Return an array of shape (len(images), embedding_dim)."""
        ...


def load_image(images_dir: Path, image_id: str) -> Image.Image:
    """Open `<images_dir>/<image_id>.jpg` as an RGB image."""
    path = Path(images_dir) / f"{image_id}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"image {image_id} not found at {path}")
    # `convert` returns a new, fully loaded image, so the file can be closed safely.
    with Image.open(path) as img:
        return img.convert("RGB")


def compute_embeddings(
    image_ids: Sequence[str],
    images_dir: Path,
    encoder: Encoder,
    batch_size: int = 32,
) -> np.ndarray:
    """Embed the given images, returning one row per image in the same order.

    Images are loaded and encoded `batch_size` at a time, so only one batch
    has to fit in (GPU) memory at once.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    image_ids = list(image_ids)
    if not image_ids:
        raise ValueError("image_ids is empty")

    batches = []
    for start in range(0, len(image_ids), batch_size):
        batch_ids = image_ids[start : start + batch_size]
        images = [load_image(images_dir, image_id) for image_id in batch_ids]

        output = np.asarray(encoder(images))
        if output.ndim != 2 or output.shape[0] != len(images):
            raise ValueError(
                f"encoder must return 2-D output with {len(images)} rows "
                f"(one per image), got shape {output.shape}"
            )
        batches.append(output)

    return np.concatenate(batches)


def save_embeddings(path: Path, image_ids: Sequence[str], embeddings: np.ndarray) -> None:
    """Save embeddings together with their image ids, so rows can't get mixed up."""
    if len(image_ids) != len(embeddings):
        raise ValueError(
            f"got {len(image_ids)} image ids but {len(embeddings)} embedding rows"
        )
    np.savez(path, image_ids=np.asarray(image_ids, dtype=str), embeddings=embeddings)


def load_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    """Load the `(image_ids, embeddings)` saved by `save_embeddings`."""
    with np.load(path) as data:
        return data["image_ids"].tolist(), data["embeddings"]
