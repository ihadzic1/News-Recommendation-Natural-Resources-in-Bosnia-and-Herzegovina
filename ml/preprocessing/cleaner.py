"""Text cleaning and NLP preprocessing utilities for Bosnian news articles."""

import re

BOSNIAN_STOPWORDS = {
    "a", "ako", "ali", "bi", "bih", "bila", "bili", "bilo", "bio", "bit",
    "biti", "bez", "blizu", "broj", "ce", "cemo", "cete", "ces", "cu",
    "da", "dakle", "dana", "danas", "dok", "do", "dva", "dvije", "gdje",
    "ga", "godina", "godine", "i", "iako", "ih", "ili", "im", "ima",
    "imaju", "imati", "ipak", "iz", "izmedju", "između", "ja", "je",
    "jedan", "jedna", "jedno", "jer", "joj", "jos", "još", "ju", "kad",
    "kada", "kako", "kao", "kaze", "kaže", "kod", "koja", "koje", "kojeg",
    "kojem", "koji", "kojih", "kojim", "kojima", "koju", "kroz", "li",
    "manje", "me", "medju", "među", "mi", "mogu", "moze", "može", "mu",
    "na", "nad", "nakon", "nam", "nas", "ne", "nego", "nema", "ni",
    "nije", "nisu", "niti", "no", "njegov", "njegova", "njegovo", "njen",
    "njena", "njeno", "njih", "njihov", "njihova", "njihovo", "o", "od",
    "oko", "on", "ona", "one", "oni", "ono", "osim", "ova", "ovaj",
    "ove", "ovi", "ovo", "pa", "pak", "po", "pod", "pored", "poslije",
    "pred", "preko", "prema", "pri", "prije", "protiv", "putem", "radi",
    "rekao", "rekla", "s", "sa", "sam", "samo", "se", "sebe", "si",
    "smo", "ste", "stoga", "stranu", "su", "sve", "svi", "svih", "svoj",
    "svoja", "svoje", "svom", "sta", "šta", "sto", "što", "ta", "tada",
    "taj", "tako", "takodje", "također", "te", "tek", "ti", "tim",
    "to", "toga", "tog", "tokom", "tome", "tu", "u", "uz", "vam", "vas",
    "vec", "već", "veoma", "vi", "vise", "više", "vrlo", "za", "zbog",
    "zato", "ze", "ce", "će", "ćemo", "ćete", "ćeš", "ću", "čak",
    "prenosi", "navodi", "istakao", "istakla", "dodao", "dodala",
    "kazao", "kazala", "izjavio", "izjavila", "naveo", "navela",
}

_NON_LETTER_RE = re.compile(r"[^a-zčćžšđ\s]")
_WHITESPACE_RE = re.compile(r"\s+")


class TextCleaner:
    """Performs lowercasing, punctuation/special-character removal,
    whitespace normalization, tokenization and Bosnian stopword removal."""

    def __init__(self, stopwords=None, min_token_length=2):
        self.stopwords = stopwords if stopwords is not None else BOSNIAN_STOPWORDS
        self.min_token_length = min_token_length

    def clean(self, text):
        """Return a cleaned lowercase string with punctuation,
        digits and special characters removed."""
        if not isinstance(text, str):
            return ""
        text = text.lower()
        text = _NON_LETTER_RE.sub(" ", text)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text

    def tokenize(self, text):
        """Split a cleaned string into a list of tokens."""
        return [t for t in self.clean(text).split() if len(t) >= self.min_token_length]

    def remove_stopwords(self, tokens):
        """Filter Bosnian stopwords from a token list."""
        return [t for t in tokens if t not in self.stopwords]

    def preprocess(self, text):
        """Full pipeline: clean -> tokenize -> stopword removal.
        Returns a single space-joined string ready for vectorization."""
        return " ".join(self.remove_stopwords(self.tokenize(text)))

    def token_set(self, text):
        """Return the set of preprocessed tokens (used for Jaccard similarity)."""
        return set(self.remove_stopwords(self.tokenize(text)))
