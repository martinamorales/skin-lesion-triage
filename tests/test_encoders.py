"""Tests for the real foundation-model encoders.

The real encoders need PyTorch and download ~350 MB models, so their tests are
skipped automatically when those libraries are not installed (locally and in
CI). They run for real in Colab, where the `embeddings` dependency group is
installed. The registry tests below run everywhere, because the encoders module
only imports PyTorch when an encoder is actually created.
"""

import numpy as np
import pytest
from PIL import Image

from skin_lesion_triage.encoders import ENCODERS, get_encoder


def test_registry_lists_both_models():
    assert set(ENCODERS) == {"dinov2", "biomedclip"}


def test_unknown_encoder_name_raises_with_the_valid_options():
    with pytest.raises(ValueError, match="dinov2"):
        get_encoder("resnet")


@pytest.fixture
def two_images() -> list[Image.Image]:
    # Different sizes on purpose: the encoder must resize them itself.
    return [Image.new("RGB", (600, 450), (180, 120, 100)), Image.new("RGB", (300, 300), "white")]


@pytest.mark.parametrize(
    ("name", "library", "dim"),
    [("dinov2", "transformers", 768), ("biomedclip", "open_clip", 512)],
)
def test_real_encoder_returns_one_finite_row_per_image(two_images, name, library, dim):
    pytest.importorskip("torch")
    pytest.importorskip(library)
    encoder = get_encoder(name)

    embeddings = encoder(two_images)

    assert embeddings.shape == (2, dim)
    assert np.isfinite(embeddings).all()
    # Same input, same output: the model is frozen and in evaluation mode.
    np.testing.assert_allclose(encoder(two_images), embeddings, rtol=1e-5, atol=1e-5)
