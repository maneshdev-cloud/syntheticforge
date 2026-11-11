# SyntheticForge

SyntheticForge is a small synthetic tabular-data factory with an evaluation
report built in. It generates new rows from a CSV, then measures both how
similar the synthetic data looks and whether individual synthetic rows are
suspiciously close to source rows.

> The bundled dataset is invented sample data. Synthetic data is not
> automatically anonymous: always run a privacy review before sharing it.

## Problem

Teams often need realistic tabular data for development, demos and testing,
but the original dataset may contain personal or commercially sensitive
records. A useful generator therefore needs two kinds of evidence:

- **Similarity:** do the generated columns, category frequencies and numeric
  correlations resemble the source?
- **Privacy diagnostics:** did the generator copy rows, or place synthetic
  rows unusually close to particular source rows?

SyntheticForge provides a transparent baseline for both questions without
requiring a deep-learning model or a hosted service.

## Features

- **Gaussian-copula generation** for numeric columns, designed to retain
  linear correlation structure while sampling through each column's empirical
  distribution.
- **Marginal sampling baseline** that samples numeric values independently
  and categorical values from observed frequencies.
- **Categorical frequency sampling** in both generation modes.
- **Similarity metrics:** Kolmogorov–Smirnov statistic for numeric columns,
  total-variation distance for categorical columns, and correlation-matrix
  differences.
- **Privacy diagnostics:** exact-match rate and a nearest-neighbour distance
  ratio comparing synthetic-to-real distances with real-to-real distances.
- **Command-line workflow** that writes a synthetic CSV and a JSON report.
- **Optional FastAPI wrapper** for programmatic use.
- **Optional SDV hook** if SDV is installed; the project works without it.

## Tech stack

- Python
- NumPy
- pandas
- SciPy
- FastAPI (optional, only for `app.py`)
- SDV (optional alternative generator)

## How to run

Install the core dependencies:

```bash
pip install -r requirements.txt
```

Run the generator on the supplied invented dataset:

```bash
python cli.py data/sample.csv \
  --output examples/sample_synthetic.csv \
  --report examples/sample_report.json \
  --method copula \
  --seed 42
```

That command produces:

- `examples/sample_synthetic.csv` — the generated dataset
- `examples/sample_report.json` — the actual metrics from the run

Use the independent marginal baseline instead:

```bash
python cli.py data/sample.csv \
  --output synthetic.csv \
  --report report.json \
  --method marginal
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

### Optional API

Install the API dependencies and start the server:

```bash
pip install fastapi uvicorn python-multipart
uvicorn app:app --reload
```

The API exposes a health endpoint and a CSV-upload generation endpoint.

## Reading the report

- A **lower KS statistic** means two numeric distributions are closer.
- A **lower total-variation distance** means category frequencies are closer.
- A **lower correlation-matrix difference** means the numeric relationship
  structure is closer.
- A **lower exact-match rate** means fewer synthetic rows are exact copies.
- The **nearest-neighbour distance ratio** compares how close synthetic rows
  are to source rows with how close source rows are to one another. It is a
  warning signal, not a formal privacy guarantee.

## Project structure

```text
syntheticforge/
├── app.py                    # Optional FastAPI wrapper
├── cli.py                    # Command-line interface
├── data/
│   └── sample.csv            # Invented sample dataset
├── evaluate.py               # Similarity and privacy metrics
├── examples/
│   ├── sample_report.json    # Metrics produced by the documented run
│   └── sample_synthetic.csv  # Synthetic output from the documented run
├── generate.py               # Marginal, Gaussian-copula and optional SDV paths
├── requirements.txt
└── tests/
    ├── test_evaluate.py
    └── test_generate.py
```

## Limitations

The Gaussian copula focuses on rank/linear dependence and empirical marginal
distributions. It does not model every nonlinear relationship, causal process,
rare event or complex constraint in a source dataset. The privacy metrics are
diagnostics, not proof that a dataset is safe to publish.

## License

MIT — see `LICENSE`.
