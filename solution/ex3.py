"""Run EM article clustering and export its evaluation tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from .article_dataset import ArticleDataset
from .cluster_evaluation import (
    export_confusion_matrix_to_csv,
    export_scores_to_csv,
    get_clusters_topics_accuracy,
)
from .em_cluster_runner import EMClusterRunner

RARE_WORD_COUNT_THRESHOLD = 3
CLUSTERS_COUNT = 9


@dataclass(frozen=True)
class RunSummary:
    """Key outputs from one complete clustering run."""

    vocabulary_size: int
    iterations: int
    accuracy: float
    scores_path: Path
    confusion_matrix_path: Path


def main(
    development_set_path: str | Path,
    topics_path: str | Path,
    output_directory: str | Path,
    *,
    max_iterations: int = 200,
    verbose: bool = True,
) -> RunSummary:
    """Cluster the supplied corpus and export score and confusion-matrix CSVs."""
    dataset = ArticleDataset(development_set_path)
    dataset.remove_rare_words(RARE_WORD_COUNT_THRESHOLD)
    word_frequency_matrix = dataset.to_word_frequency_matrix()
    vocabulary_size = word_frequency_matrix.shape[1]
    if verbose:
        print(f"Scanned source dataset; retained {vocabulary_size} words")

    runner = EMClusterRunner(
        dataset.articles_info,
        CLUSTERS_COUNT,
        word_frequency_matrix,
    )
    if verbose:
        print("Finished preparing; starting EM clustering")
    scores = runner.run(max_iterations=max_iterations, verbose=verbose)
    article_clusters = runner.get_clusters()

    with Path(topics_path).open(encoding="utf-8") as topics_file:
        topics = tuple(line.strip() for line in topics_file if line.strip())
    if len(topics) != CLUSTERS_COUNT:
        raise ValueError(f"expected {CLUSTERS_COUNT} topics, received {len(topics)}")

    destination = Path(output_directory)
    scores_path = export_scores_to_csv(scores, destination / "model-scores.csv")
    matrix_path = export_confusion_matrix_to_csv(
        article_clusters,
        topics,
        destination / "confusion-matrix.csv",
    )
    accuracy = get_clusters_topics_accuracy(article_clusters)
    if verbose:
        print(f"Clustering accuracy: {accuracy}")

    return RunSummary(
        vocabulary_size=vocabulary_size,
        iterations=len(scores),
        accuracy=accuracy,
        scores_path=scores_path,
        confusion_matrix_path=matrix_path,
    )


def cli() -> None:
    """Parse command-line arguments and run the clustering workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("development_set", type=Path)
    parser.add_argument("topics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("run-results"))
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    for file_path in (args.development_set, args.topics):
        if not file_path.is_file():
            parser.error(f"file does not exist: {file_path}")

    main(
        args.development_set,
        args.topics,
        args.output_dir,
        max_iterations=args.max_iterations,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    cli()
