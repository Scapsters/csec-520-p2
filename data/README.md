# Data

**Do not commit large datasets.** This directory is git-ignored except its documentation and the instructor-provided small Iris dataset in `iris/`.

## Default: Iris (no download needed)

`config.yaml` ships with `data.source: iris`, which loads the Iris dataset
bundled with scikit-learn. Use it to debug your from-scratch implementation:
150 points, 4 features, 3 well-understood classes.

## Instructor-provided Iris files

The course Google Drive copy is included in [`iris/`](iris/README.md), both in its original form and as a CSV with headers. Run it with:

```bash
.venv/bin/python -m src.cluster --config configs/iris-local.yaml
```

This copy has historical differences at two records compared with scikit-learn's bundled version; see `iris/README.md`. The default configuration is unchanged.

## Project dataset: UNSW-NB15

The deliverable uses a realistic network-traffic dataset.

1. Download the [instructor-provided dataset archive from Google Drive](https://drive.google.com/file/d/1jDsXYALnEsLzYYGCEtxgkAYygS-vKIt7/view?usp=sharing)
   (about 20.6 GB).
2. Extract the **partitioned training CSV**, `UNSW_NB15_training-set.csv`,
   from the archive. Use the training CSV for the primary experiment, not raw
   packet captures or the full collection of CSVs.
3. Place it at `data/UNSW_NB15_training-set.csv` (it stays untracked).
4. Point `config.yaml` at the file:

```yaml
data:
  source: csv
  csv_path: data/UNSW_NB15_training-set.csv
  target: attack_cat          # labels: EVALUATION ONLY
  drop_columns: [id, label]   # id is an index; label duplicates attack_cat
  subsample: 4000
```

### Notes

- `attack_cat` is the 10-class label; `label` is the binary 0/1 version. Drop
  whichever you are not evaluating against — keeping both leaks the target.
- Categorical columns (`proto`, `service`, `state`) are dropped automatically by
  the numeric-only selector. Encoding them is a reasonable extension — say so in
  your report if you do.
- A from-scratch K-means is comfortable on a few thousand rows, not millions.
  Keep `subsample` modest; it is stratified by class.

### Citation

Moustafa, N., & Slay, J. (2015). *UNSW-NB15: A Comprehensive Data Set for Network
Intrusion Detection Systems.* Military Communications and Information Systems
Conference (MilCIS), IEEE.

## Optional mirror fallback and checksum

Use the Google Drive archive above as the primary course download. If it is
unavailable, `make data` downloads the partitioned training CSV from this
optional public mirror (after `make setup`):
https://github.com/Nir-J/ML-Projects/blob/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv
The original dataset source is
https://research.unsw.edu.au/projects/unsw-nb15-dataset.
The mirror is not publisher-authenticated; this checksum identifies the tested
copy (175,341 rows, 45 columns including ID and labels):

```
bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa
```

The downloader checks this SHA-256 before installing the file and refuses to
replace an existing mismatched file. When the CSV is already present, `make data` only verifies it against this
checksum. The Drive archive contents have not been checksum-verified against
this mirror; a mismatch should be investigated rather than replacing the file. An alternate URL serving identical bytes can be passed with
`python -m src.download_data --url URL`. A different legitimate dataset version
requires documenting its provenance and updating the checksum deliberately.
Acquisition belongs to setup; reproduction does not download data.

The balanced cap of 4,000 yields 3,730 observations for this file: 400 in nine
classes and all 130 Worms. Label-based balancing is explicitly permitted, but
labels and their duplicates must never be clustering features. Report actual
counts and recognize the resulting change in class prevalence.
