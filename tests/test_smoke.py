"""Smoke test: checks that the package is installed and importable."""

import skin_lesion_triage


def test_package_has_version():
    assert skin_lesion_triage.__version__ == "0.1.0"
