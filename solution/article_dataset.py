"""
Helper for minimizing reading from the dataset files and parsing them
"""
from collections import Counter
from dataclasses import dataclass, field
from itertools import count
from pathlib import Path
from typing import Counter as CounterType, Dict, List, Optional, Set, Tuple

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ArticleUnigramInfo:
    """
    Dataclass for parsed article unigram information.
    Contains the fields:
    * ordinal_id (id by order in the dataset)
    * text (the article's text)
    * words_counter (counter for the article's words)
    * total_words (the total number of words in the article)
    * article_topics (the topics of the article according to the dataset)
    """
    text: str
    words_counter: CounterType[str]
    total_words: int
    article_topics: Set[str]
    ordinal_id: int = field(default_factory=count().__next__)

    @staticmethod
    def from_text_header(text: str, header: str) -> 'ArticleUnigramInfo':
        """
        Parse the article from its text and header
        """
        text_words = text.split()
        text_words_counter = Counter(text_words)
        article_topics = set(header.strip()[1:-1].split()[2:])
        return ArticleUnigramInfo(
            text=text,
            words_counter=text_words_counter,
            total_words=sum(text_words_counter.values()),
            article_topics=article_topics
        )

    def __hash__(self):
        return hash(self.text)


class ArticleDataset:
    """
    Helper class for reading and parsing the given dataset files
    """

    def __init__(self, dataset_file_path: Path):
        """
        :param dataset_file_path: file path for the dataset
        """
        self._dataset_file_processed = False
        self._dataset_file_path = dataset_file_path
        self._dataset_file = open(dataset_file_path, "r")

        self._word_to_id: Dict[str, int] = {}
        self._articles_info: Optional[Tuple[ArticleUnigramInfo, ...]] = None
        self._words_counter: Counter = Counter()

    def _process_dataset_file(self):
        """
        Iterator for the words in the articles as presented in the dataset file
        :return: word in the article_info
        """
        if self._dataset_file_processed:
            return
        articles_info: List[ArticleUnigramInfo] = []
        article_header = ""

        self._dataset_file.seek(0)  # start from the beginning of the file
        for line_num, line in enumerate(self._dataset_file):
            if line_num % 4 == 0:  # article header
                article_header = line
            elif line_num % 4 == 2:  # article content
                articles_info.append(ArticleUnigramInfo.from_text_header(line, article_header))

        self._words_counter: CounterType[str] = Counter()
        for article_info in articles_info:
            self._words_counter.update(article_info.words_counter)
        for word_id, word in enumerate(self._words_counter):
            self._word_to_id[word] = word_id
        self._articles_info = tuple(articles_info)
        self._dataset_file_processed = True

    def remove_rare_words(self, count_threshold: int):
        """
        Removes rare words from the dataset, by count threshold
        """

        def _remove_word_from_articles(to_remove: str):
            for article_info in self._articles_info:
                if to_remove in article_info.words_counter:
                    article_info.words_counter.pop(to_remove)

        for word, word_cnt in self.words_counter.items():
            if word_cnt <= count_threshold:
                _remove_word_from_articles(word)
                self._words_counter.pop(word)

        self._word_to_id = {}
        for word_id, word in enumerate(self._words_counter):
            self._word_to_id[word] = word_id

    def to_word_frequency_matrix(self) -> npt.NDArray[np.integer]:
        """
        Exports the dataset as word to frequency matrix
        """
        if not self._dataset_file_processed:
            self._process_dataset_file()

        word_frequency_matrix = np.zeros((len(self._articles_info), len(self._words_counter)))
        for article_info in self._articles_info:
            for word in self._words_counter:
                word_frequency_matrix[article_info.ordinal_id, self._word_to_id[word]] = (
                    article_info.words_counter[word]
                )
        return word_frequency_matrix

    @property
    def articles_info(self) -> Tuple[ArticleUnigramInfo, ...]:
        """
        Information on all the articles in the dataset
        """
        if not self._dataset_file_processed:
            self._process_dataset_file()

        return self._articles_info

    @property
    def words_counter(self) -> Counter:
        """
        Global words counter for the dataset
        """
        if not self._dataset_file_processed:
            self._process_dataset_file()

        return self._words_counter.copy()
