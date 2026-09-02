"""
Logic for the em clustering algorithm
"""
from dataclasses import dataclass
from math import exp, log
from typing import Dict, List, Set, Tuple

import numpy as np
import numpy.typing as npt

from article_dataset import ArticleUnigramInfo

ALPHA_EPSILON = 0.001
SMOOTHING_FACTOR = 0.061
K_VALUE = 10


@dataclass(order=True)
class LanguageModelScore:
    """
    Dataclass for the scores of a language model
    """
    ln_likelihood: float
    mean_word_perplexity: float


# noinspection PyTestUnpassedFixture
class EMClusterRunner:
    """
    Runner for the em clustering algorithm
    """
    MINIMUM_ITERATIONS = 5  # Minimum iterations for the clustering, to let it rev up
    STOP_LN_LIKELIHOOD_THRESHOLD = 5  # Algorithm stop threshold based on ln-likelihood score

    def __init__(self,
                 articles: Tuple[ArticleUnigramInfo, ...],
                 clusters_cnt: int,
                 word_freq_matrix: npt.NDArray[np.integer]):
        """
        :param articles: information about the articles to do clustering on
        :param clusters_cnt: number of clusters
        :param word_freq_matrix: global word to frequency matrix
        """
        self._articles = articles
        self._cluster_cnt = clusters_cnt
        self._article_word_freq = word_freq_matrix
        self._total_words = word_freq_matrix.shape[1]
        self._articles_total_words = np.sum(self._article_word_freq, axis=1)

        self._cluster_article_probabilities: npt.NDArray[np.float64] = np.zeros((clusters_cnt, len(articles)))
        self._init_article_clustering()

        self._clusters_alpha: npt.NDArray[np.float64] = np.zeros((clusters_cnt,))
        self._update_clusters_alpha()
        self._clusters_p_values: npt.NDArray[np.float64] = np.zeros((clusters_cnt, self._total_words))
        self._update_clusters_p_values()
        self._article_ln_likelihood_cache: Dict[ArticleUnigramInfo, float] = {}
        self._model_scores: List[LanguageModelScore] = []

    def _init_article_clustering(self):
        """
        Initializes the clustering by their ordinal id
        """
        for article in self._articles:
            initial_article_cluster = article.ordinal_id % self._cluster_cnt
            article_id = article.ordinal_id
            self._cluster_article_probabilities[initial_article_cluster, article_id] = 1.0

    def _should_stop_run(self) -> bool:
        """
        Checks if the clustering algorithm should stop
        """
        if not len(self._model_scores) > self.MINIMUM_ITERATIONS:
            return False
        last_likelihood_distance = abs(self._model_scores[-1].ln_likelihood - self._model_scores[-2].ln_likelihood)
        prior_likelihood_distance = abs(self._model_scores[-2].ln_likelihood - self._model_scores[-3].ln_likelihood)
        return (last_likelihood_distance < self.STOP_LN_LIKELIHOOD_THRESHOLD and
                prior_likelihood_distance < self.STOP_LN_LIKELIHOOD_THRESHOLD)

    def run(self):
        """
        Run the em clustering algorithm
        """
        while not self._should_stop_run():
            self._e_step()
            self._m_step()

            ln_likelihood = self._get_ln_likelihood()
            mean_perplexity = self._get_perplexity()
            self._model_scores.append(LanguageModelScore(ln_likelihood, mean_perplexity))
            self._article_ln_likelihood_cache = {}

            print(f"Iteration {len(self._model_scores): 3}: "
                  f"ln-likelihood = {ln_likelihood: 20}, mean perplexity = {mean_perplexity: 20}")

    def _update_clusters_alpha(self):
        """
        Updates the clusters' alpha values
        """
        clusters_probabilities: List[float] = []
        for cluster_id in range(self._cluster_cnt):
            cluster_probabilities_sum = np.sum(self._cluster_article_probabilities[cluster_id])
            cluster_alpha = max((1 / len(self._articles)) * cluster_probabilities_sum, ALPHA_EPSILON)
            clusters_probabilities.append(cluster_alpha)
        total_sum = sum(clusters_probabilities)
        self._clusters_alpha = np.fromiter((cluster_probability / total_sum
                                            for cluster_probability in clusters_probabilities), np.float64)

    def _update_clusters_p_values(self):
        """
        Updates the clusters' p values
        """
        for cluster_id in range(self._cluster_cnt):
            cluster_probabilities = self._cluster_article_probabilities[cluster_id]
            normalize_value = (
                    np.sum(cluster_probabilities * self._articles_total_words) +
                    self._total_words * SMOOTHING_FACTOR
            )

            word_cluster_probability = (np.sum(cluster_probabilities * self._article_word_freq.transpose(), axis=1)
                                        + SMOOTHING_FACTOR)
            self._clusters_p_values[cluster_id] = word_cluster_probability / normalize_value

    def _m_step(self):
        """
        M-step in the EM clustering algorithm
        """
        self._update_clusters_alpha()
        self._update_clusters_p_values()

    def _get_clusters_z_values(self, article: ArticleUnigramInfo) -> npt.NDArray[np.float64]:
        """
        Return the z values (as defined in the exercise) of the clusters
        """
        alpha_factor = np.log(self._clusters_alpha)
        p_values_factor = np.sum(self._article_word_freq[article.ordinal_id] * np.log(self._clusters_p_values),
                                 axis=1)
        return alpha_factor + p_values_factor

    def _e_step(self):
        """
        E-step in the EM clustering algorithm
        """
        for article in self._articles:
            article_z_values = self._get_clusters_z_values(article)
            max_z_value = np.max(article_z_values)
            lowered_z_values = article_z_values - max_z_value

            below_threshold_mask = lowered_z_values < -1 * K_VALUE
            above_threshold_mask = np.logical_not(below_threshold_mask)
            lowered_z_values[below_threshold_mask] = 0
            lowered_z_values[above_threshold_mask] = (np.exp(lowered_z_values[above_threshold_mask]) /
                                                      np.sum(np.exp(lowered_z_values[above_threshold_mask])))

            self._cluster_article_probabilities[:, article.ordinal_id] = lowered_z_values

    def _get_article_ln_likelihood(self, article: ArticleUnigramInfo) -> float:
        """
        Returns article's ln likelihood in the model, it is cached in same iteration
        """

        def _inner_get_article_ln_likelihood(likelihood_article: ArticleUnigramInfo) -> float:
            z_values = self._get_clusters_z_values(likelihood_article)
            max_z_value = np.max(z_values)
            lowered_z_values = z_values - max_z_value

            above_threshold_mask = lowered_z_values >= -1 * K_VALUE
            return max_z_value + log(np.sum(np.exp(lowered_z_values[above_threshold_mask])))

        if article not in self._article_ln_likelihood_cache:
            self._article_ln_likelihood_cache[article] = _inner_get_article_ln_likelihood(article)
        return self._article_ln_likelihood_cache[article]

    def _get_ln_likelihood(self) -> float:
        """
        Returns sum of ln-likelihoods of all articles
        """
        ln_likelihood_value: float = 0
        for article in self._articles:
            ln_likelihood_value += self._get_article_ln_likelihood(article)
        return ln_likelihood_value

    def _get_perplexity(self) -> float:
        """
        Returns the perplexity of the clustering model
        """
        perplexity_sum = 0
        for article in self._articles:
            perplexity_sum += self._get_article_ln_likelihood(article)

        return exp(perplexity_sum * -1 / np.sum(self._articles_total_words))

    def get_clusters(self) -> Tuple[Set[ArticleUnigramInfo], ...]:
        """
        Return the clusters made by the model
        """
        clusters: List[Set[ArticleUnigramInfo]] = [set() for _ in range(self._cluster_cnt)]
        for article in self._articles:
            article_likely_cluster = np.argmax(self._cluster_article_probabilities[:, article.ordinal_id])
            clusters[article_likely_cluster].add(article)

        return tuple(clusters)

    def get_cluster_runner_model_scores(self) -> Tuple[LanguageModelScore, ...]:
        """
        Returns the model's score history
        """
        return tuple(self._model_scores)
