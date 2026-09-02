import csv
from pathlib import Path

import pytest

from solution.ex3 import main

REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.mark.integration
def test_complete_corpus_reproduces_reference_results(tmp_path) -> None:
    summary = main(
        REPOSITORY_ROOT / "data" / "develop.txt",
        REPOSITORY_ROOT / "data" / "topics.txt",
        tmp_path,
        verbose=False,
    )

    assert summary.vocabulary_size == 6800
    assert summary.iterations == 33
    assert summary.accuracy == pytest.approx(0.6332391713747646)
    assert summary.confusion_matrix_path.read_text(encoding="utf-8") == (
        REPOSITORY_ROOT / "results" / "confusion-matrix.csv"
    ).read_text(encoding="utf-8")

    with summary.scores_path.open(encoding="utf-8", newline="") as actual_file:
        actual_rows = list(csv.DictReader(actual_file))
    with (REPOSITORY_ROOT / "results" / "model-scores.csv").open(
        encoding="utf-8", newline=""
    ) as expected_file:
        expected_rows = list(csv.DictReader(expected_file))

    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows, strict=True):
        assert actual["iteration number"] == expected["iteration number"]
        assert float(actual["ln likelihood score"]) == pytest.approx(
            float(expected["ln likelihood score"]), rel=1e-10
        )
        assert float(actual["perplexity score"]) == pytest.approx(
            float(expected["perplexity score"]), rel=1e-10
        )
