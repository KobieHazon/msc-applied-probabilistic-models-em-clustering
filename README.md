# EM Article Clustering

A 2024 CS MSc Applied Probabilistic Models exercise implementing expectation-maximization clustering from the underlying equations. The project groups Reuters-derived articles into nine unigram language-model clusters, handles probability underflow in log space, tracks likelihood and perplexity, and evaluates the final hard assignments against supplied topic labels.

## Tech Stack

- Python 3.10 or newer
- NumPy for dense matrix operations and probability calculations
- Python CSV tooling for deterministic evaluation exports
- pytest and Ruff for validation and code quality
- uv for reproducible dependency and package management

## Setup

```bash
git clone https://github.com/KobieHazon/msc-applied-probabilistic-models-em-clustering.git
cd msc-applied-probabilistic-models-em-clustering
uv sync --dev
```

## Usage

Run the complete recovered experiment:

```bash
uv run apm-em-cluster data/develop.txt data/topics.txt
```

The command writes `model-scores.csv` and `confusion-matrix.csv` under `run-results/`. On the supplied corpus, the reference run retains a 6,800-word vocabulary, converges after 33 iterations, and reaches classification accuracy `0.6332391713747646`.

Use `--output-dir`, `--max-iterations`, or `--quiet` to control generated output, the convergence safety bound, and progress logging.

## Testing

Run the fast deterministic suite:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

Run the complete corpus regression separately:

```bash
uv run pytest -m integration
```

The integration check compares all 33 likelihood and perplexity records, the final confusion matrix, vocabulary size, and accuracy with the preserved results. Small floating-point differences between Python and NumPy versions are tolerated; cluster assignments and reported conclusions must remain unchanged.

## Repository Structure

- `assignment/`: supplied exercise brief and underflow/scaling background note
- `data/`: supplied Reuters-derived development corpus and topic list
- `solution/`: my NumPy EM implementation and evaluation tools
- `results/`: canonical score and confusion-matrix CSVs reproduced from the solution
- `report.pdf`: authored analysis, plots, histograms, and reported conclusions
- `tests/`: focused tests and the opt-in full-corpus regression

## Implementation notes

The maintained version adds deterministic output paths, independent article numbering across dataset instances, bounded EM execution, faster frequency-matrix construction, packaging, tests, and documentation.
