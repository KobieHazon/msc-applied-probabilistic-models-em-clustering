from solution.article_dataset import ArticleDataset

CORPUS = (
    "<TRAIN 1 acq>\n\ncommon alpha rare\n\n"
    "<TRAIN 2 money-fx>\n\ncommon beta\n\n"
    "<TRAIN 3 acq money-fx>\n\ncommon alpha\n\n"
)


def test_dataset_parsing_filtering_and_matrix_creation(tmp_path) -> None:
    corpus_path = tmp_path / "develop.txt"
    corpus_path.write_text(CORPUS, encoding="utf-8")
    dataset = ArticleDataset(corpus_path)

    dataset.remove_rare_words(1)
    matrix = dataset.to_word_frequency_matrix()

    assert [article.ordinal_id for article in dataset.articles_info] == [0, 1, 2]
    assert dataset.articles_info[2].article_topics == frozenset({"acq", "money-fx"})
    assert dataset.words_counter == {"common": 3, "alpha": 2}
    assert matrix.shape == (3, 2)
    assert matrix.sum(axis=1).tolist() == [2.0, 1.0, 2.0]


def test_article_ordinals_restart_for_each_dataset(tmp_path) -> None:
    corpus_path = tmp_path / "develop.txt"
    corpus_path.write_text(CORPUS, encoding="utf-8")

    first = ArticleDataset(corpus_path)
    second = ArticleDataset(corpus_path)

    assert [article.ordinal_id for article in first.articles_info] == [0, 1, 2]
    assert [article.ordinal_id for article in second.articles_info] == [0, 1, 2]
