# Data

This folder is ignored by Git (except this file). Datasets are downloaded on demand
and are **never** committed to the repository.

## HAM10000

10,015 dermatoscopic images of pigmented skin lesions across 7 diagnostic categories,
with metadata (lesion id, diagnosis, diagnosis method, age, sex, body location).

Download it from the repository root with:

```bash
uv run python -m skin_lesion_triage.download
```

This fetches ~2.8 GB from [Harvard Dataverse](https://doi.org/10.7910/DVN/DBW86T),
verifies the MD5 checksum of every file and produces:

```
data/ham10000/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1.zip
├── HAM10000_images_part_2.zip
└── images/              # 10,015 .jpg files
```

Re-running the command is safe: files that are already present and valid are skipped.

**License:** [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)
(non-commercial use only; redistribution is not allowed).

**Citation:**

> Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection
> of multi-source dermatoscopic images of common pigmented skin lesions.
> *Sci. Data* 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
