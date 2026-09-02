from collections import Counter

import numpy as np

from solution.article_dataset import ArticleUnigramInfo
from solution.em_cluster_runner import EMClusterRunner


def test_em_converges_and_assigns_every_article() -> None:
    matrix = np.array(
        [
            [5.0, 0.0],
            [4.0, 1.0],
            [0.0, 5.0],
            [1.0, 4.0],
        ]
    )
    articles = tuple(
        ArticleUnigramInfo(
            ordinal_id=index,
            text=f"article {index}",
            words_counter=Counter(),
            total_words=int(matrix[index].sum()),
            article_topics=frozenset({"left" if index < 2 else "right"}),
        )
        for index in range(4)
    )
    runner = EMClusterRunner(articles, 2, matrix)

    scores = runner.run(max_iterations=50, verbose=False)
    clusters = runner.get_clusters()

    assert len(scores) >= 6
    assert all(np.isfinite(score.ln_likelihood) for score in scores)
    assert all(np.isfinite(score.mean_word_perplexity) for score in scores)
    assert sum(len(cluster) for cluster in clusters) == len(articles)
    assert {article.ordinal_id for cluster in clusters for article in cluster} == set(range(4))
