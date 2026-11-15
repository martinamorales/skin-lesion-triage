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

## Notebooks

The analysis is documented step by step in Jupyter notebooks. Each one can be opened
directly in Google Colab, where it downloads the data and installs the project itself.

| Notebook | Description | |
|---|---|---|
| [01 · Data exploration](notebooks/01_data_exploration.ipynb) | Class imbalance, label quality, images per lesion (data leakage), patient demographics | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/martinamorales/skin-lesion-triage/blob/main/notebooks/01_data_exploration.ipynb) |

To run them locally instead: `uv sync --group notebooks`, then open the notebook in
VS Code or Jupyter.

## Data

Datasets are **not** included in this repository. See [`data/README.md`](data/README.md)
for download instructions.

## Disclaimer

This is a research and educational project. It is **not** a medical device and must not
be used for clinical diagnosis.

## License

[MIT](LICENSE)
