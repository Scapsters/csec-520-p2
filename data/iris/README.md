# Course Iris copy

Source: the instructor's `Iris.7z` in the course Google Drive Project 2 folder:
https://drive.google.com/file/d/1Hvu4awRQO9wRufiLK7v0Cc0IfJfXd2mt/view

- `iris.data`: original headerless data, preserved byte-for-byte.
- `iris.names`: original dataset description and attribution, preserved byte-for-byte.
- `iris.csv`: the same 150 records with column headers for the template CSV loader.

The four measurements are sepal length, sepal width, petal length, and petal width in centimeters. The `species` column has 50 observations per class and is used for evaluation only. The original description attributes the data to R. A. Fisher and documents historical data discrepancies.

This copy differs from scikit-learn's bundled Iris at records 35 and 38 (one-based). It has been preserved as supplied, so exact metrics can differ from the default `data.source: iris` run.

From the repository root, after `make setup`, run:

```bash
.venv/bin/python -m src.cluster --config configs/iris-local.yaml
```

Outputs go to `results/iris-local/`. To make this the `make reproduce` dataset, copy `configs/iris-local.yaml` to `config.yaml`. The default config continues to use scikit-learn's bundled Iris. This small teaching dataset is intentionally tracked; large datasets such as UNSW-NB15 remain ignored.
