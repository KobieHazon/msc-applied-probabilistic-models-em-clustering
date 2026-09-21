"""Evaluate clusters and export the report tables."""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from .article_dataset import ArticleUnigramInfo
from .em_cluster_runner import LanguageModelScore

SCORES_CSV_FIELD_NAMES = (
    "iteration number",
    "ln likelihood score",
    "perplexity score",
)


def export_scores_to_csv(scores: Iterable[LanguageModelScore], output_path: str | Path) -> Path:
    """Export per-iteration likelihood and perplexity scores."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as scores_csv:
        writer = csv.writer(scores_csv)
        writer.writerow(SCORES_CSV_FIELD_NAMES)
        writer.writerows(
            (iteration, score.ln_likelihood, score.mean_word_perplexity)
            for iteration, score in enumerate(scores)
        )
    return destination


def get_cluster_topic_confusion_matrix(
    clusters: Sequence[Collection[ArticleUnigramInfo]],
    topics: Sequence[str],
) -> tuple[tuple[int, ...], ...]:
    """Return the cluster-by-topic confusion matrix."""
    topic_to_column = {topic: index for index, topic in enumerate(topics)}
    matrix = [[0 for _ in topics] for _ in clusters]

    for cluster_row, cluster in enumerate(clusters):
        for article in cluster:
            for topic in article.article_topics:
                try:
                    topic_column = topic_to_column[topic]
                except KeyError as exc:
                    raise ValueError(f"article contains unknown topic: {topic}") from exc
                matrix[cluster_row][topic_column] += 1
    return tuple(tuple(row) for row in matrix)


def export_confusion_matrix_to_csv(
    clusters: Sequence[Collection[ArticleUnigramInfo]],
    topics: Sequence[str],
    output_path: str | Path,
) -> Path:
    """Export a size-sorted cluster-by-topic confusion matrix."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = get_cluster_topic_confusion_matrix(clusters, topics)
    rows = sorted(enumerate(matrix), key=lambda item: sum(item[1]), reverse=True)

    with destination.open("w", encoding="utf-8", newline="") as matrix_csv:
        writer = csv.writer(matrix_csv)
        writer.writerow((r"cluster number\topic", *topics, "cluster size"))
        writer.writerows((cluster_id, *counts, sum(counts)) for cluster_id, counts in rows)
    return destination


def _get_cluster_topic(cluster: Collection[ArticleUnigramInfo]) -> str:
    if not cluster:
        raise ValueError("an empty cluster has no dominant topic")
    topic_counts = Counter(topic for article in cluster for topic in article.article_topics)
    return topic_counts.most_common(1)[0][0]


def get_clusters_topics_accuracy(
    clusters: Sequence[Collection[ArticleUnigramInfo]],
) -> float:
    """Return dominant-topic assignment accuracy across nonempty clusters."""
    correct_assignments = 0
    total_assignments = 0
    for cluster in clusters:
        if not cluster:
            continue
        cluster_topic = _get_cluster_topic(cluster)
        correct_assignments += sum(cluster_topic in article.article_topics for article in cluster)
        total_assignments += len(cluster)

    if total_assignments == 0:
        raise ValueError("accuracy requires at least one clustered article")
    return correct_assignments / total_assignments
