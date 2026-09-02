"""Main flow of the exercise, calls the different stages and does validation for the invocation"""
import os
from pathlib import Path
from sys import argv

from article_dataset import ArticleDataset
from cluster_evaluation import export_confusion_matrix_to_csv, export_scores_to_csv, get_clusters_topics_accuracy
from em_cluster_runner import EMClusterRunner

RARE_WORD_COUNT_THRESHOLD = 3  # below this threshold, a word is "rare"
CLUSTERS_COUNT = 9  # the number of clusters to do clustering on


def main(development_set_path: Path, topics_path: Path):
    """
    Main flow for the exercise, calls the different stages.
    :param development_set_path: file name for the development set
    :param topics_path:
    """
    articles_dataset = ArticleDataset(development_set_path)
    articles_dataset.remove_rare_words(RARE_WORD_COUNT_THRESHOLD)
    word_frequency_matrix = articles_dataset.to_word_frequency_matrix()
    print(f"Scanned the source dataset, the vocabulary size is {word_frequency_matrix.shape[1]} words")
    em_cluster_runner = EMClusterRunner(articles_dataset.articles_info, CLUSTERS_COUNT, word_frequency_matrix)
    print(f"Finished preparing for clustering, starting now...")
    em_cluster_runner.run()
    print(f"Finished Clustering, exporting results")
    article_clusters = em_cluster_runner.get_clusters()

    with open(topics_path, "r") as topics_file:
        topics = tuple(line.strip() for line in topics_file.readlines() if line.strip())
    export_scores_to_csv(em_cluster_runner.get_cluster_runner_model_scores())
    export_confusion_matrix_to_csv(article_clusters, topics)
    print(f"Clustering Accuracy: {get_clusters_topics_accuracy(article_clusters)}")


if __name__ == "__main__":
    _, development_set_file_path, topics_file_path = argv
    for file_path in (development_set_file_path, topics_file_path):
        if not os.path.isfile(file_path):
            raise ValueError(f"file {file_path} does not exist")

    main(development_set_file_path, topics_file_path)
