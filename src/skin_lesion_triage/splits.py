"""Lesion-level train/val/test splits for HAM10000.

HAM10000 contains several images of the same lesion (different angles,
zoom levels or dates). If images of one lesion end up in both train and test,
the model can "recognise" the lesion instead of learning to diagnose it, and
the test score becomes optimistic. This is called data leakage.

To avoid it we split *lesions*, not images: every image inherits the split of
its lesion. The split is also stratified by diagnosis, so rare classes
(e.g. dermatofibroma) appear in every split in roughly the same proportion.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = ("lesion_id", "dx")


def _validate(metadata: pd.DataFrame, val_frac: float, test_frac: float) -> None:
    """Raise ValueError if the inputs cannot produce a sensible split."""
    missing = [col for col in REQUIRED_COLUMNS if col not in metadata.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")

    if not (0 < val_frac < 1 and 0 < test_frac < 1):
        raise ValueError("val_frac and test_frac must both be between 0 and 1 (exclusive)")
    if val_frac + test_frac >= 1:
        raise ValueError("val_frac + test_frac must be < 1 so that train is not empty")

    # Stratifying by lesion only makes sense if each lesion has a single diagnosis.
    dx_per_lesion = metadata.groupby("lesion_id")["dx"].nunique()
    inconsistent = dx_per_lesion[dx_per_lesion > 1].index.tolist()
    if inconsistent:
        raise ValueError(f"lesions with more than one diagnosis: {inconsistent[:5]}")


def make_splits(
    metadata: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign every image to train, val or test, keeping lesions together.

    Args:
        metadata: One row per image, with at least `lesion_id` and `dx` columns.
        val_frac: Fraction of *lesions* assigned to validation.
        test_frac: Fraction of *lesions* assigned to test.
        seed: Random seed, so the same inputs always give the same split.

    Returns:
        A copy of `metadata` with an extra `split` column. The input is not modified.
    """
    _validate(metadata, val_frac, test_frac)

    # One row per lesion: this is the unit we split.
    lesions = metadata.drop_duplicates("lesion_id")[["lesion_id", "dx"]]

    # Step 1: separate the test lesions from the rest.
    rest, test = train_test_split(
        lesions, test_size=test_frac, stratify=lesions["dx"], random_state=seed
    )
    # Step 2: split the rest into train and val. The fraction is rescaled because
    # `rest` is smaller than the full set: e.g. 0.15 / 0.85 ≈ 0.176 of `rest`.
    train, val = train_test_split(
        rest,
        test_size=val_frac / (1 - test_frac),
        stratify=rest["dx"],
        random_state=seed,
    )

    split_of_lesion = pd.concat(
        [
            pd.Series("train", index=train["lesion_id"]),
            pd.Series("val", index=val["lesion_id"]),
            pd.Series("test", index=test["lesion_id"]),
        ]
    )

    result = metadata.copy()
    result["split"] = result["lesion_id"].map(split_of_lesion)
    return result
