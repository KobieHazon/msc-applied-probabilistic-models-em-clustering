"""Expectation-maximization clustering for article unigram models."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np
import numpy.typing as npt

from .article_dataset import ArticleUnigramInfo

ALPHA_EPSILON = 0.001
SMOOTHING_FACTOR = 0.061
UNDERFLOW_THRESHOLD = 10


@dataclass(frozen=True, order=True)
class LanguageModelScore:
    """Log-likelihood and mean per-word perplexity for one iteration."""

    ln_likelihood: float
    mean_word_perplexity: float


class EMClusterRunner:
    """Cluster articles with multinomial mixture-model EM."""

    MINIMUM_ITERATIONS = 5
    STOP_LN_LIKELIHOOD_THRESHOLD = 5

    def __init__(
        self,
        articles: tuple[ArticleUnigramInfo, ...],
        clusters_count: int,
        word_frequency_matrix: npt.NDArray[np.float64],
    ):
        if not articles:
            raise ValueError("EM requires at least one article")
        if not 0 < clusters_count <= len(articles):
            raise ValueError("cluster count must be between one and the article count")
        if word_frequency_matrix.shape[0] != len(articles):
            raise ValueError("frequency-matrix rows must match the article count")
        if word_frequency_matrix.shape[1] == 0:
            raise ValueError("frequency matrix must contain at least one word")
        if tuple(article.ordinal_id for article in articles) != tuple(range(len(articles))):
            raise ValueError("article ordinal IDs must be contiguous and zero-based")

        self._articles = articles
        self._cluster_count = clusters_count
        self._article_word_frequency = word_frequency_matrix
        self._vocabulary_size = word_frequency_matrix.shape[1]
        self._articles_total_words = np.sum(self._article_word_frequency, axis=1)

        self._cluster_article_probabilities = np.zeros(
            (clusters_count, len(articles)), dtype=np.float64
        )
        self._initialize_article_clustering()

        self._clusters_alpha = np.zeros((clusters_count,), dtype=np.float64)
        self._update_clusters_alpha()
        self._clusters_p_values = np.zeros(
            (clusters_count, self._vocabulary_size), dtype=np.float64
        )
        self._update_clusters_p_values()
        self._article_ln_likelihood_cache: dict[ArticleUnigramInfo, float] = {}
        self._model_scores: list[LanguageModelScore] = []

    def _initialize_article_clustering(self) -> None:
        for article in self._articles:
            initial_cluster = article.ordinal_id % self._cluster_count
            self._cluster_article_probabilities[initial_cluster, article.ordinal_id] = 1.0

    def _should_stop_run(self) -> bool:
        if len(self._model_scores) <= self.MINIMUM_ITERATIONS:
            return False
        last_distance = abs(
            self._model_scores[-1].ln_likelihood - self._model_scores[-2].ln_likelihood
        )
        prior_distance = abs(
            self._model_scores[-2].ln_likelihood - self._model_scores[-3].ln_likelihood
        )
        return (
            last_distance < self.STOP_LN_LIKELIHOOD_THRESHOLD
            and prior_distance < self.STOP_LN_LIKELIHOOD_THRESHOLD
        )

    def run(
        self, max_iterations: int = 200, *, verbose: bool = True
    ) -> tuple[LanguageModelScore, ...]:
        """Run EM until convergence or fail at the configured safety bound."""
        if max_iterations <= self.MINIMUM_ITERATIONS:
            raise ValueError(f"max_iterations must exceed {self.MINIMUM_ITERATIONS}")

        while not self._should_stop_run():
            if len(self._model_scores) >= max_iterations:
                raise RuntimeError(f"EM did not converge within {max_iterations} iterations")

            self._e_step()
            self._m_step()
            score = LanguageModelScore(
                self._get_ln_likelihood(),
                self._get_perplexity(),
            )
            self._model_scores.append(score)
            self._article_ln_likelihood_cache.clear()

            if verbose:
                print(
                    f"Iteration {len(self._model_scores):3}: "
                    f"ln-likelihood = {score.ln_likelihood:20}, "
                    f"mean perplexity = {score.mean_word_perplexity:20}"
                )

        return self.model_scores

    def _update_clusters_alpha(self) -> None:
        probabilities = []
        for cluster_id in range(self._cluster_count):
            cluster_sum = np.sum(self._cluster_article_probabilities[cluster_id])
            probabilities.append(max(cluster_sum / len(self._articles), ALPHA_EPSILON))
        total = sum(probabilities)
        self._clusters_alpha = np.fromiter(
            (probability / total for probability in probabilities),
            dtype=np.float64,
            count=self._cluster_count,
        )

    def _update_clusters_p_values(self) -> None:
        for cluster_id in range(self._cluster_count):
            cluster_probabilities = self._cluster_article_probabilities[cluster_id]
            normalizer = (
                np.sum(cluster_probabilities * self._articles_total_words)
                + self._vocabulary_size * SMOOTHING_FACTOR
            )
            word_cluster_probability = (
                np.sum(
                    cluster_probabilities * self._article_word_frequency.transpose(),
                    axis=1,
                )
                + SMOOTHING_FACTOR
            )
            self._clusters_p_values[cluster_id] = word_cluster_probability / normalizer

    def _m_step(self) -> None:
        self._update_clusters_alpha()
        self._update_clusters_p_values()

    def _get_clusters_z_values(self, article: ArticleUnigramInfo) -> npt.NDArray[np.float64]:
        alpha_factor = np.log(self._clusters_alpha)
        p_values_factor = np.sum(
            self._article_word_frequency[article.ordinal_id] * np.log(self._clusters_p_values),
            axis=1,
        )
        return alpha_factor + p_values_factor

    def _e_step(self) -> None:
        for article in self._articles:
            z_values = self._get_clusters_z_values(article)
            lowered_z_values = z_values - np.max(z_values)
            active = lowered_z_values >= -UNDERFLOW_THRESHOLD
            probabilities = np.zeros_like(lowered_z_values)
            exponentials = np.exp(lowered_z_values[active])
            probabilities[active] = exponentials / np.sum(exponentials)
            self._cluster_article_probabilities[:, article.ordinal_id] = probabilities

    def _get_article_ln_likelihood(self, article: ArticleUnigramInfo) -> float:
        if article not in self._article_ln_likelihood_cache:
            z_values = self._get_clusters_z_values(article)
            max_z_value = np.max(z_values)
            lowered_z_values = z_values - max_z_value
            active = lowered_z_values >= -UNDERFLOW_THRESHOLD
            self._article_ln_likelihood_cache[article] = max_z_value + log(
                np.sum(np.exp(lowered_z_values[active]))
            )
        return self._article_ln_likelihood_cache[article]

    def _get_ln_likelihood(self) -> float:
        return sum(self._get_article_ln_likelihood(article) for article in self._articles)

    def _get_perplexity(self) -> float:
        likelihood = sum(self._get_article_ln_likelihood(article) for article in self._articles)
        return exp(-likelihood / np.sum(self._articles_total_words))

    def get_clusters(self) -> tuple[set[ArticleUnigramInfo], ...]:
        """Return the hard assignment for every article."""
        clusters = [set() for _ in range(self._cluster_count)]
        for article in self._articles:
            likely_cluster = int(
                np.argmax(self._cluster_article_probabilities[:, article.ordinal_id])
            )
            clusters[likely_cluster].add(article)
        return tuple(clusters)

    @property
    def model_scores(self) -> tuple[LanguageModelScore, ...]:
        """Return the score history accumulated so far."""
        return tuple(self._model_scores)
