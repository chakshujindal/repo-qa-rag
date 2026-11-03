from repoqa.text import tokenize


def test_splits_snake_and_camel_case_and_keeps_the_whole_identifier() -> None:
    assert tokenize("parseHttpRequest") == ["parsehttprequest", "parse", "http", "request"]
    assert tokenize("max_file_bytes") == ["max_file_byte", "max", "file", "byte"]


def test_drops_stopwords_single_chars_and_folds_plurals() -> None:
    assert tokenize("How are the chunks split? x") == ["chunk", "split"]


def test_keeps_double_s_words_and_numbers() -> None:
    assert tokenize("class 404") == ["class", "404"]
