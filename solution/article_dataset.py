"""Parse the Reuters-derived article corpus used by the clustering exercise."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


@dataclass(eq=False)
class ArticleUnigramInfo:
    """Unigram information and reference topics for one article."""

    ordinal_id: int
    text: str
    words_counter: Counter[str]
    total_words: int
    article_topics: frozenset[str]

    @classmethod
    def from_text_header(
        cls,
        text: str,
        header: str,
        ordinal_id: int,
    ) -> ArticleUnigramInfo:
        """Parse one article body and its supplied Reuters header."""
        header_fields = header.strip()[1:-1].split()
        if len(header_fields) < 3:
            raise ValueError(f"invalid article header: {header!r}")

        words_counter = Counter(text.split())
        return cls(
            ordinal_id=ordinal_id,
            text=text,
            words_counter=words_counter,
            total_words=sum(words_counter.values()),
            article_topics=frozenset(header_fields[2:]),
        )


class ArticleDataset:
    """Load articles and produce the dense frequency matrix used by EM."""

    def __init__(self, dataset_file_path: str | Path):
        self._dataset_file_path = Path(dataset_file_path)
        self._dataset_file_processed = False
        self._word_to_id: dict[str, int] = {}
        self._articles_info: tuple[ArticleUnigramInfo, ...] | None = None
        self._words_counter: Counter[str] = Counter()

    def _process_dataset_file(self) -> None:
        if self._dataset_file_processed:
            return

        with self._dataset_file_path.open(encoding="utf-8") as dataset_file:
            lines = [line.strip() for line in dataset_file if line.strip()]

        if len(lines) % 2:
            raise ValueError("dataset must contain a header and body for every article")

        articles = tuple(
            ArticleUnigramInfo.from_text_header(
                header=lines[index],
                text=lines[index + 1],
                ordinal_id=index // 2,
            )
            for index in range(0, len(lines), 2)
        )
        if not articles:
            raise ValueError("dataset contains no articles")

        self._words_counter = Counter()
        for article in articles:
            self._words_counter.update(article.words_counter)

        self._articles_info = articles
        self._rebuild_word_ids()
        self._dataset_file_processed = True

    def _rebuild_word_ids(self) -> None:
        self._word_to_id = {word: word_id for word_id, word in enumerate(self._words_counter)}

    def remove_rare_words(self, count_threshold: int) -> None:
        """Remove words whose global count is at or below the threshold."""
        if count_threshold < 0:
            raise ValueError("rare-word threshold cannot be negative")
        self._process_dataset_file()
        assert self._articles_info is not None

        rare_words = {
            word for word, count in self._words_counter.items() if count <= count_threshold
        }
        for article in self._articles_info:
            for word in rare_words & article.words_counter.keys():
                del article.words_counter[word]
            article.total_words = sum(article.words_counter.values())

        for word in rare_words:
            del self._words_counter[word]
        self._rebuild_word_ids()

    def to_word_frequency_matrix(self) -> npt.NDArray[np.float64]:
        """Return an article-by-word frequency matrix."""
        self._process_dataset_file()
        assert self._articles_info is not None

        matrix = np.zeros((len(self._articles_info), len(self._words_counter)), dtype=np.float64)
        for article in self._articles_info:
            for word, count in article.words_counter.items():
                word_id = self._word_to_id.get(word)
                if word_id is not None:
                    matrix[article.ordinal_id, word_id] = count
        return matrix

    @property
    def articles_info(self) -> tuple[ArticleUnigramInfo, ...]:
        """Information for every parsed article in source order."""
        self._process_dataset_file()
        assert self._articles_info is not None
        return self._articles_info

    @property
    def words_counter(self) -> Counter[str]:
        """A copy of the global corpus word counter."""
        self._process_dataset_file()
        return self._words_counter.copy()
