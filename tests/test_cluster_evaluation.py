import csv
from collections import Counter

from solution.article_dataset import ArticleUnigramInfo
from solution.cluster_evaluation import (
    export_confusion_matrix_to_csv,
    export_scores_to_csv,
    get_cluster_topic_confusion_matrix,
    get_clusters_topics_accuracy,
)
from solution.em_cluster_runner import LanguageModelScore


def article(ordinal_id: int, *topics: str) -> ArticleUnigramInfo:
    return ArticleUnigramInfo(
        ordinal_id=ordinal_id,
        text=f"article {ordinal_id}",
        words_counter=Counter({"word": 1}),
        total_words=1,
        article_topics=frozenset(topics),
    )


def test_confusion_matrix_and_dominant_topic_accuracy() -> None:
    clusters = ({article(0, "acq"), article(1, "acq", "money-fx")}, {article(2, "grain")})
    topics = ("acq", "money-fx", "grain")

    assert get_cluster_topic_confusion_matrix(clusters, topics) == (
        (2, 1, 0),
        (0, 0, 1),
    )
    assert get_clusters_topics_accuracy(clusters) == 1.0


def test_csv_exports_are_deterministic(tmp_path) -> None:
    clusters = ({article(0, "acq"), article(1, "acq")}, {article(2, "grain")})
    scores = (LanguageModelScore(-10.0, 4.0), LanguageModelScore(-8.0, 3.0))

    scores_path = export_scores_to_csv(scores, tmp_path / "scores.csv")
    matrix_path = export_confusion_matrix_to_csv(
        clusters, ("acq", "grain"), tmp_path / "matrix.csv"
    )

    with scores_path.open(encoding="utf-8", newline="") as score_file:
        assert list(csv.reader(score_file)) == [
            ["iteration number", "ln likelihood score", "perplexity score"],
            ["0", "-10.0", "4.0"],
            ["1", "-8.0", "3.0"],
        ]
    with matrix_path.open(encoding="utf-8", newline="") as matrix_file:
        assert list(csv.reader(matrix_file)) == [
            ["cluster number\\topic", "acq", "grain", "cluster size"],
            ["0", "2", "0", "2"],
            ["1", "0", "1", "1"],
        ]
