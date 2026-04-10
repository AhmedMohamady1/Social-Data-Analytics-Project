import argparse
from collections import Counter
import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
import symspellpy
from symspellpy import SymSpell, Verbosity

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NUMBER_RE = re.compile(r"\d+")
WHITESPACE_RE = re.compile(r"\s+")

EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]+",
    flags=re.UNICODE,
)

PUNCT_RE = re.compile(r"[.!?,;:]+")
TOKEN_RE = re.compile(r"__exclaim__|__question__|[A-Za-z_]+(?:'[A-Za-z]+)?")
FINAL_PUNCT_RE = re.compile(r"[^\w\s]")
ELONGATION_RE = re.compile(r"(.)\1{2,}")

BASE_PROTECTED_WORDS = {
    "iran",
    "israel",
    "gaza",
    "hormuz",
    "netanyahu",
    "tehran",
    "usa",
    "uk",
    "epstein",
    "qatar",
    "uae",
    "houthi",
    "hezbollah",
    "hamas",
    "cnn",
    "bbc",
    "mastodon",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}

NEGATION_WORDS = {"no", "not", "never", "cannot", "neither"}
BOOSTER_WORDS = {"very", "extremely", "really", "highly", "super", "too"}
HEDGE_WORDS = {"maybe", "perhaps", "possibly", "likely", "unclear", "allegedly"}
MODAL_WORDS = {"might", "could", "should", "must", "may", "would"}

TOKEN_CANONICAL_MAP = {
    "us": "usa",
    "u": "usa",
    "uk": "united_kingdom",
    "uae": "united_arab_emirates",
    "teheran": "tehran",
    "isreal": "israel",
    "hezbullah": "hezbollah",
    "iranwar": "iran_war",
    "israelwar": "israel_war",
    "eupol": "eu_politics",
    "uspol": "us_politics",
    "epsteinfiles": "epstein_file",
}

BIGRAM_PHRASE_MAP = {
    ("middle", "east"): "middle_east",
    ("white", "house"): "white_house",
    ("oil", "price"): "oil_price",
    ("supreme", "leader"): "supreme_leader",
    ("foreign", "minister"): "foreign_minister",
    ("united", "states"): "usa",
    ("war", "crime"): "war_crime",
    ("regime", "change"): "regime_change",
}

TRIGRAM_PHRASE_MAP = {
    ("united", "arab", "emirates"): "united_arab_emirates",
    ("islamic", "revolutionary", "guard"): "irgc",
    ("strait", "of", "hormuz"): "hormuz_strait",
}

TOPIC_TERMS = {
    "conflict": {
        "war",
        "strike",
        "attack",
        "missile",
        "drone",
        "airstrike",
        "retaliation",
        "irgc",
    },
    "diplomacy": {
        "ceasefire",
        "negotiation",
        "sanction",
        "agreement",
        "diplomacy",
        "embassy",
    },
    "energy": {
        "oil",
        "gas",
        "hormuz",
        "hormuz_strait",
        "shipping",
        "barrel",
    },
    "media": {
        "cnn",
        "bbc",
        "headline",
        "report",
        "clip",
        "thread",
        "video",
    },
}

ENTITY_TERMS = {
    "iran": {"iran", "tehran", "irgc", "khamenei"},
    "israel": {"israel", "idf", "netanyahu", "tel_aviv"},
    "usa": {"usa", "white_house", "pentagon", "washington"},
    "gulf": {"qatar", "uae", "united_arab_emirates", "hormuz_strait"},
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "shouldn't": "should not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "mustn't": "must not",
    "i'm": "i am",
    "it's": "it is",
    "they're": "they are",
    "we're": "we are",
    "you're": "you are",
}


class PipelineCNormalizer:
    def __init__(
        self,
        max_edit_distance: int,
        min_token_length: int,
        extra_protected_words: set[str],
        negation_window: int,
        hedge_window: int,
        modal_window: int,
        remove_stopwords: bool,
        strip_non_ascii: bool,
        discard_hashtags: bool,
        disable_topic_tags: bool,
        disable_entity_tags: bool,
        disable_discourse_tags: bool,
    ) -> None:
        self.max_edit_distance = max_edit_distance
        self.min_token_length = min_token_length
        self.negation_window = max(0, negation_window)
        self.hedge_window = max(0, hedge_window)
        self.modal_window = max(0, modal_window)
        self.remove_stopwords = remove_stopwords
        self.strip_non_ascii = strip_non_ascii
        self.discard_hashtags = discard_hashtags
        self.disable_topic_tags = disable_topic_tags
        self.disable_entity_tags = disable_entity_tags
        self.disable_discourse_tags = disable_discourse_tags
        self.protected_words = BASE_PROTECTED_WORDS.union(extra_protected_words)

        self.sym_spell = SymSpell(
            max_dictionary_edit_distance=max_edit_distance,
            prefix_length=7,
        )

        sym_path = Path(symspellpy.__file__).parent
        dictionary_path = sym_path / "frequency_dictionary_en_82_765.txt"
        loaded = self.sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)
        if not loaded:
            raise RuntimeError(f"Could not load SymSpell dictionary: {dictionary_path}")

    @staticmethod
    def reduce_elongation(token: str) -> str:
        # Example: sooooo -> soo
        return ELONGATION_RE.sub(r"\1\1", token)

    def correct_token(self, token: str) -> str:
        token = token.lower()
        token = self.reduce_elongation(token)

        if token in self.protected_words:
            return token
        if len(token) < self.min_token_length:
            return token

        suggestions = self.sym_spell.lookup(
            token,
            Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance,
            include_unknown=True,
        )
        if not suggestions:
            return token

        return suggestions[0].term.lower()

    @staticmethod
    def expand_contractions(text: str) -> str:
        for contraction, expansion in CONTRACTIONS.items():
            text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def canonicalize_tokens(tokens: list[str]) -> list[str]:
        return [TOKEN_CANONICAL_MAP.get(token, token) for token in tokens]

    @staticmethod
    def base_token(token: str) -> str:
        for prefix in ("neg_", "int_", "hedge_", "modal_"):
            if token.startswith(prefix):
                return token[len(prefix):]
        return token

    @staticmethod
    def join_domain_phrases(tokens: list[str]) -> list[str]:
        output = []
        i = 0
        while i < len(tokens):
            if i + 2 < len(tokens):
                triplet = (tokens[i], tokens[i + 1], tokens[i + 2])
                if triplet in TRIGRAM_PHRASE_MAP:
                    output.append(TRIGRAM_PHRASE_MAP[triplet])
                    i += 3
                    continue

            if i + 1 < len(tokens):
                pair = (tokens[i], tokens[i + 1])
                if pair in BIGRAM_PHRASE_MAP:
                    output.append(BIGRAM_PHRASE_MAP[pair])
                    i += 2
                    continue
            output.append(tokens[i])
            i += 1
        return output

    @staticmethod
    def remove_adjacent_duplicates(tokens: list[str]) -> list[str]:
        if not tokens:
            return tokens

        deduped = [tokens[0]]
        for token in tokens[1:]:
            if token != deduped[-1]:
                deduped.append(token)
        return deduped

    def apply_intensity_scope(self, tokens: list[str]) -> list[str]:
        scoped = []
        boost_next = False

        for token in tokens:
            if token in BOOSTER_WORDS:
                scoped.append(token)
                boost_next = True
                continue

            if boost_next and token not in {"exclamation", "question"}:
                scoped.append(f"int_{self.base_token(token)}")
                boost_next = False
                continue

            scoped.append(token)

        return scoped

    def apply_negation_scope(self, tokens: list[str]) -> list[str]:
        scoped = []
        remaining_scope = 0

        for token in tokens:
            if token in NEGATION_WORDS:
                scoped.append(token)
                remaining_scope = self.negation_window
                continue

            if token in {"exclamation", "question"}:
                scoped.append(token)
                continue

            if remaining_scope > 0:
                scoped.append(f"neg_{self.base_token(token)}")
                remaining_scope -= 1
            else:
                scoped.append(token)

        return scoped

    def apply_hedge_scope(self, tokens: list[str]) -> list[str]:
        if self.hedge_window <= 0:
            return tokens

        scoped = []
        remaining_scope = 0

        for token in tokens:
            root = self.base_token(token)
            if root in HEDGE_WORDS:
                scoped.append(root)
                remaining_scope = self.hedge_window
                continue

            if root in {"exclamation", "question"}:
                scoped.append(root)
                continue

            if remaining_scope > 0:
                scoped.append(f"hedge_{root}")
                remaining_scope -= 1
            else:
                scoped.append(token)

        return scoped

    def apply_modal_scope(self, tokens: list[str]) -> list[str]:
        if self.modal_window <= 0:
            return tokens

        scoped = []
        remaining_scope = 0

        for token in tokens:
            root = self.base_token(token)
            if root in MODAL_WORDS:
                scoped.append(root)
                remaining_scope = self.modal_window
                continue

            if root in {"exclamation", "question"}:
                scoped.append(root)
                continue

            if remaining_scope > 0:
                scoped.append(f"modal_{root}")
                remaining_scope -= 1
            else:
                scoped.append(token)

        return scoped

    def inject_topic_tags(self, tokens: list[str]) -> list[str]:
        if self.disable_topic_tags:
            return tokens

        token_roots = {self.base_token(token) for token in tokens}
        tags = [
            f"topic_{topic}"
            for topic, terms in TOPIC_TERMS.items()
            if token_roots.intersection(terms)
        ]
        return tokens + tags

    def inject_entity_tags(self, tokens: list[str]) -> list[str]:
        if self.disable_entity_tags:
            return tokens

        token_roots = {self.base_token(token) for token in tokens}
        tags = [
            f"entity_{entity}"
            for entity, terms in ENTITY_TERMS.items()
            if token_roots.intersection(terms)
        ]
        return tokens + tags

    def inject_discourse_tags(self, tokens: list[str]) -> list[str]:
        if self.disable_discourse_tags:
            return tokens

        token_roots = [self.base_token(token) for token in tokens]
        tags = []
        if "question" in token_roots:
            tags.append("discourse_interrogative")
        if "exclamation" in token_roots:
            tags.append("discourse_emphasis")
        if any(root in HEDGE_WORDS for root in token_roots):
            tags.append("discourse_uncertainty")
        if any(root in MODAL_WORDS for root in token_roots):
            tags.append("discourse_modal")
        return tokens + tags

    def preprocess(self, text: str) -> str:
        text = "" if pd.isna(text) else str(text)
        text = unicodedata.normalize("NFKC", html.unescape(text))
        text = self.expand_contractions(text)

        # Keep punctuation markers here so boundaries remain explicit during tokenization.
        text = URL_RE.sub(" ", text)
        text = MENTION_RE.sub(" ", text)
        if self.discard_hashtags:
            text = HASHTAG_RE.sub(" ", text)
        else:
            text = HASHTAG_RE.sub(lambda m: f" {m.group(1)} ", text)
        text = HTML_TAG_RE.sub(" ", text)
        text = EMOJI_RE.sub(" ", text)
        text = NUMBER_RE.sub(" ", text)
        if self.strip_non_ascii:
            text = text.encode("ascii", errors="ignore").decode("ascii")

        text = re.sub(r"!+", " __exclaim__ ", text)
        text = re.sub(r"\?+", " __question__ ", text)
        text = PUNCT_RE.sub(" ", text)

        raw_tokens = TOKEN_RE.findall(text)

        corrected_tokens: list[str] = []
        for token in raw_tokens:
            token = token.lower()
            if token == "__exclaim__":
                corrected_tokens.append("exclamation")
                continue
            if token == "__question__":
                corrected_tokens.append("question")
                continue
            corrected_tokens.append(self.correct_token(token))

        corrected_tokens = self.canonicalize_tokens(corrected_tokens)
        corrected_tokens = self.join_domain_phrases(corrected_tokens)
        corrected_tokens = self.apply_intensity_scope(corrected_tokens)
        corrected_tokens = self.apply_negation_scope(corrected_tokens)
        corrected_tokens = self.apply_hedge_scope(corrected_tokens)
        corrected_tokens = self.apply_modal_scope(corrected_tokens)

        if self.remove_stopwords:
            corrected_tokens = [
                token for token in corrected_tokens if token not in STOPWORDS or token in NEGATION_WORDS
            ]

        corrected_tokens = self.remove_adjacent_duplicates(corrected_tokens)
        corrected_tokens = self.inject_topic_tags(corrected_tokens)
        corrected_tokens = self.inject_entity_tags(corrected_tokens)
        corrected_tokens = self.inject_discourse_tags(corrected_tokens)

        text = " ".join(token for token in corrected_tokens if token)
        text = FINAL_PUNCT_RE.sub(" ", text)
        text = text.lower()
        text = WHITESPACE_RE.sub(" ", text).strip()
        return text


def apply_corpus_frequency_filter(series: pd.Series, min_corpus_freq: int) -> pd.Series:
    if min_corpus_freq <= 1:
        return series

    token_counts: Counter[str] = Counter()
    for text in series.fillna(""):
        token_counts.update(text.split())

    keep_tokens = {token for token, count in token_counts.items() if count >= min_corpus_freq}
    return series.apply(
        lambda text: " ".join(token for token in str(text).split() if token in keep_tokens)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline C: Context-aware lexical, semantic, and discourse normalization for final_text. "
            "Removes hashtags/mentions/URLs/numbers, normalizes whitespace, "
            "applies SymSpell correction with protected domain vocabulary, then "
            "adds contraction expansion, phrase canonicalization, negation/intensity/hedge/modal scope, "
            "injects topic/entity/discourse tags, "
            "and optional corpus-frequency filtering."
        )
    )
    parser.add_argument("--input", type=str, required=True, help="Input CSV path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV path")
    parser.add_argument(
        "--text_column",
        type=str,
        default="final_text",
        help="Column to preprocess (default: final_text)",
    )
    parser.add_argument(
        "--output_column",
        type=str,
        default="final_text_style_c",
        help="Name of processed text column",
    )
    parser.add_argument(
        "--max_edit_distance",
        type=int,
        default=2,
        help="SymSpell max edit distance",
    )
    parser.add_argument(
        "--min_token_length",
        type=int,
        default=3,
        help="Minimum token length to consider for correction",
    )
    parser.add_argument(
        "--protected_words",
        type=str,
        default="",
        help="Comma-separated words to skip correction",
    )
    parser.add_argument(
        "--negation_window",
        type=int,
        default=2,
        help="How many tokens after a negator get prefixed with neg_",
    )
    parser.add_argument(
        "--hedge_window",
        type=int,
        default=1,
        help="How many tokens after a hedge term get prefixed with hedge_",
    )
    parser.add_argument(
        "--modal_window",
        type=int,
        default=1,
        help="How many tokens after a modal verb get prefixed with modal_",
    )
    parser.add_argument(
        "--remove_stopwords",
        action="store_true",
        help="Remove generic stopwords after correction",
    )
    parser.add_argument(
        "--strip_non_ascii",
        action="store_true",
        help="Drop non-ASCII characters after text normalization",
    )
    parser.add_argument(
        "--discard_hashtags",
        action="store_true",
        help="Drop hashtag terms entirely instead of keeping hashtag text",
    )
    parser.add_argument(
        "--disable_topic_tags",
        action="store_true",
        help="Disable topic_* tag injection",
    )
    parser.add_argument(
        "--disable_entity_tags",
        action="store_true",
        help="Disable entity_* tag injection",
    )
    parser.add_argument(
        "--disable_discourse_tags",
        action="store_true",
        help="Disable discourse_* tag injection",
    )
    parser.add_argument(
        "--min_corpus_freq",
        type=int,
        default=1,
        help="Drop tokens occurring fewer than this count across all rows",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        raise ValueError(
            f"Missing text column '{args.text_column}'. Available columns: {list(df.columns)}"
        )

    extra_protected = {
        word.strip().lower()
        for word in args.protected_words.split(",")
        if word.strip()
    }

    normalizer = PipelineCNormalizer(
        max_edit_distance=args.max_edit_distance,
        min_token_length=args.min_token_length,
        extra_protected_words=extra_protected,
        negation_window=args.negation_window,
        hedge_window=args.hedge_window,
        modal_window=args.modal_window,
        remove_stopwords=args.remove_stopwords,
        strip_non_ascii=args.strip_non_ascii,
        discard_hashtags=args.discard_hashtags,
        disable_topic_tags=args.disable_topic_tags,
        disable_entity_tags=args.disable_entity_tags,
        disable_discourse_tags=args.disable_discourse_tags,
    )

    df[args.output_column] = df[args.text_column].apply(normalizer.preprocess)
    df[args.output_column] = apply_corpus_frequency_filter(
        df[args.output_column],
        min_corpus_freq=args.min_corpus_freq,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    empty_count = int((df[args.output_column].str.strip() == "").sum())
    changed_rows = int((df[args.output_column] != df[args.text_column].fillna("")).sum())
    print("Pipeline C complete.")
    print(f"Input rows: {len(df)}")
    print(f"Output saved to: {output_path}")
    print(f"Output column: {args.output_column}")
    print(f"Empty processed rows: {empty_count}")
    print(f"Rows changed vs source column '{args.text_column}': {changed_rows}")


if __name__ == "__main__":
    main()
