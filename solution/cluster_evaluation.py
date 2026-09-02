"""
Utils for evaluating the clustering model's performance
"""
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Collection, Iterable, List, Set, Tuple

from article_dataset import ArticleUnigramInfo
from em_cluster_runner import LanguageModelScore

_SCORES_CSV_FIELD_NAMES = ["iteration number", "ln likelihood score", "perplexity score"]
_CLUSTER_EVALUATION_DIRECTORY: Path = Path("cluster_evaluation")
_EVALUATION_TIME = datetime.now()


def export_scores_to_csv(scores: Iterable[LanguageModelScore]):
    """
    Exports the model's score to a csv
    """

    if not _CLUSTER_EVALUATION_DIRECTORY.is_dir():
        _CLUSTER_EVALUATION_DIRECTORY.mkdir()
    with open(_CLUSTER_EVALUATION_DIRECTORY / f'model_scores_{_EVALUATION_TIME.strftime("%Y-%m-%d-%H-%M-%S")}.csv',
              'w', newline='') as scores_csv:
        scores_writer = csv.DictWriter(scores_csv, fieldnames=_SCORES_CSV_FIELD_NAMES)

        scores_writer.writeheader()
        scores_writer.writerows([
            {
                _SCORES_CSV_FIELD_NAMES[0]: iteration_num,
                _SCORES_CSV_FIELD_NAMES[1]: score.ln_likelihood,
                _SCORES_CSV_FIELD_NAMES[2]: score.mean_word_perplexity
            }
            for iteration_num, score in enumerate(scores)
        ])


def _get_cluster_topic_confusion_matrix(
        clusters: Collection[Set[ArticleUnigramInfo]],
        topics: Collection[str]
) -> Tuple[List[int], ...]:
    """
    Returns the cluster-topic confusion matrix
    """
    topic_to_column_num = {topic: topic_num for topic_num, topic in enumerate(topics)}
    confusion_matrix: Tuple[List[int], ...] = tuple([0 for _ in range(len(topics))] for _ in range(len(clusters)))

    for cluster_row, cluster in enumerate(clusters):
        for article in cluster:
            for topic in article.article_topics:
                topic_column = topic_to_column_num[topic]
                confusion_matrix[cluster_row][topic_column] += 1
    return confusion_matrix


def export_confusion_matrix_to_csv(clusters: Collection[Set[ArticleUnigramInfo]], topics: Collection[str]):
    """
    Exports a confusion matrix of the article's clustering to a csv
    """
    if not _CLUSTER_EVALUATION_DIRECTORY.is_dir():
        _CLUSTER_EVALUATION_DIRECTORY.mkdir()

    confusion_matrix = _get_cluster_topic_confusion_matrix(clusters, topics)
    confusion_matrix_csv_field_names = [r"cluster number\topic"] + [topic for topic in topics] + ["cluster size"]
    with open(_CLUSTER_EVALUATION_DIRECTORY / f'confusion_matrix_{_EVALUATION_TIME.strftime("%Y-%m-%d-%H-%M-%S")}.csv',
              'w', newline='') as confusion_matrix_csv:
        confusion_matrix_writer = csv.DictWriter(confusion_matrix_csv, fieldnames=confusion_matrix_csv_field_names)

        confusion_matrix_writer.writeheader()
        confusion_matrix_writer.writerows(sorted([
            {
                confusion_matrix_csv_field_names[0]: cluster_number,
                **{confusion_matrix_csv_field_names[topic_num + 1]: confusion_matrix[cluster_number][topic_num]
                   for topic_num in range(len(topics))},
                confusion_matrix_csv_field_names[-1]: sum(confusion_matrix[cluster_number])
            }
            for cluster_number, cluster_confusion in enumerate(confusion_matrix)
        ], key=lambda row: row[confusion_matrix_csv_field_names[-1]], reverse=True))


def _get_cluster_topic(cluster: Set[ArticleUnigramInfo]) -> str:
    """
    Return the topic our clustering model gave to a cluster
    """
    all_topics = [topic for article in cluster for topic in article.article_topics]
    cluster_topics_counter = Counter(all_topics)
    return cluster_topics_counter.most_common(1)[0][0]


def get_clusters_topics_accuracy(clusters: Tuple[Set[ArticleUnigramInfo], ...]):
    """
    Returns the accuracy of the clustering model
    """
    correct_assignments = 0
    total_assignments = 0
    for cluster in clusters:
        cluster_topic = _get_cluster_topic(cluster)
        correct_assignments += sum(cluster_topic in article.article_topics for article in cluster)
        total_assignments += len(cluster)
    return correct_assignments / total_assignments
