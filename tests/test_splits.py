"""Tests for lesion-level train/val/test splitting.

These tests use a small synthetic metadata table that mimics HAM10000
(several images per lesion, imbalanced classes), so they run in CI without
the real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from skin_lesion_triage.splits import SPLIT_NAMES, make_splits

# Number of lesions per diagnosis: imbalanced on purpose, like HAM10000.
LESIONS_PER_CLASS = {
    "nv": 200,
    "mel": 60,
    "bkl": 60,
    "bcc": 40,
    "akiec": 30,
    "vasc": 20,
    "df": 20,
}


@pytest.fixture
def metadata() -> pd.DataFrame:
    """Synthetic metadata: each lesion has 1-4 images sharing its diagnosis."""
    rng = np.random.default_rng(0)
    rows = []
    lesion_counter = 0
    image_counter = 0
    for dx, n_lesions in LESIONS_PER_CLASS.items():
        for _ in range(n_lesions):
            lesion_id = f"HAM_{lesion_counter:07d}"
            lesion_counter += 1
            for _ in range(rng.integers(1, 5)):
                rows.append(
                    {"lesion_id": lesion_id, "image_id": f"ISIC_{image_counter:07d}", "dx": dx}
                )
                image_counter += 1
    return pd.DataFrame(rows)


def test_every_image_gets_exactly_one_valid_split(metadata):
    result = make_splits(metadata)

    assert len(result) == len(metadata)
    assert result["image_id"].is_unique
    assert set(result["image_id"]) == set(metadata["image_id"])
    assert set(result["split"]) == set(SPLIT_NAMES)


def test_no_lesion_appears_in_more_than_one_split(metadata):
    # The core anti-leakage guarantee: all images of a lesion stay together.
    result = make_splits(metadata)

    splits_per_lesion = result.groupby("lesion_id")["split"].nunique()
    assert (splits_per_lesion == 1).all()


def test_every_diagnosis_is_present_in_every_split(metadata):
    result = make_splits(metadata)

    for split in SPLIT_NAMES:
        diagnoses = set(result.loc[result["split"] == split, "dx"])
        assert diagnoses == set(LESIONS_PER_CLASS), f"missing classes in {split}"


def test_split_sizes_are_close_to_requested_fractions(metadata):
    result = make_splits(metadata, val_frac=0.15, test_frac=0.15)

    lesions = result.drop_duplicates("lesion_id")
    fractions = lesions["split"].value_counts(normalize=True)
    assert fractions["train"] == pytest.approx(0.70, abs=0.05)
    assert fractions["val"] == pytest.approx(0.15, abs=0.05)
    assert fractions["test"] == pytest.approx(0.15, abs=0.05)


def test_same_seed_gives_same_split(metadata):
    first = make_splits(metadata, seed=123)
    second = make_splits(metadata, seed=123)

    pd.testing.assert_frame_equal(first, second)


def test_different_seeds_give_different_splits(metadata):
    first = make_splits(metadata, seed=1)
    second = make_splits(metadata, seed=2)

    assert not first["split"].equals(second["split"])


def test_input_dataframe_is_not_modified(metadata):
    original = metadata.copy()

    make_splits(metadata)

    pd.testing.assert_frame_equal(metadata, original)


def test_missing_required_column_raises(metadata):
    with pytest.raises(ValueError, match="lesion_id"):
        make_splits(metadata.drop(columns="lesion_id"))


@pytest.mark.parametrize(
    ("val_frac", "test_frac"),
    [(0.0, 0.15), (0.15, 0.0), (0.5, 0.5), (-0.1, 0.2)],
)
def test_invalid_fractions_raise(metadata, val_frac, test_frac):
    with pytest.raises(ValueError):
        make_splits(metadata, val_frac=val_frac, test_frac=test_frac)
