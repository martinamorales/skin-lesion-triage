# Skin Lesion Triage

> An uncertainty-aware skin lesion classifier that knows when to defer to a dermatologist.

[![CI](https://github.com/martinamorales/skin-lesion-triage/actions/workflows/ci.yml/badge.svg)](https://github.com/martinamorales/skin-lesion-triage/actions/workflows/ci.yml)

**Status:** 🚧 Work in progress

## Motivation

Most skin lesion classifiers report a single accuracy number. In a clinical setting, a
model that is confidently wrong is dangerous. This project explores a more realistic
workflow: the model handles the cases it is confident about and **refers uncertain
cases to a human expert**.

## Planned features

- [ ] Patient/lesion-level data splits (no data leakage)
- [ ] Baseline classifier on top of pretrained foundation-model embeddings
- [ ] Probability calibration (reliability diagrams, temperature scaling)
- [ ] Selective prediction: risk–coverage curves and a "defer" option
- [ ] Evaluation under distribution shift and across skin-tone subgroups
- [ ] Interactive demo and model card

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/martinamorales/skin-lesion-triage.git
cd skin-lesion-triage
uv sync          # create the virtual environment and install dependencies
uv run pytest    # run the test suite
```

## Data

Datasets are **not** included in this repository. See [`data/README.md`](data/README.md)
for download instructions.

## Disclaimer

This is a research and educational project. It is **not** a medical device and must not
be used for clinical diagnosis.

## License

[MIT](LICENSE)
