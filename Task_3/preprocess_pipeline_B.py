import argparse
from collections import Counter
import html
import re
import string
import unicodedata
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
MENTION_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NUMBER_RE = re.compile(r"\d+")
WHITESPACE_RE = re.compile(r"\s+")
EXCLAIM_RE = re.compile(r"!+")
QUESTION_RE = re.compile(r"\?+")
ELONGATION_RE = re.compile(r"(.)\1{2,}")

# Covers common emoji blocks and pictographs.
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
    "u",
    "us",
    "was",
    "were",
    "with",
    "you",
    "your",
}

NEGATION_WORDS = {"no", "not", "never"}
BOOSTER_WORDS = {"very", "extremely", "really", "highly", "super", "too", "so"}

TOKEN_CANONICAL_MAP = {
    "us": "usa",
    "u": "usa",
    "uk": "united_kingdom",
    "uae": "united_arab_emirates",
    "teheran": "tehran",
    "hezbullah": "hezbollah",
    "isreal": "israel",
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
        "bomb",
        "military",
        "airstrike",
        "retaliation",
    },
    "diplomacy": {
        "talk",
        "ceasefire",
        "sanction",
        "negotiation",
        "diplomacy",
        "agreement",
        "embassy",
    },
    "energy": {
        "oil",
        "gas",
        "hormuz",
        "hormuz_strait",
        "barrel",
        "price",
        "shipping",
    },
    "media": {
        "cnn",
        "bbc",
        "headline",
        "video",
        "clip",
        "report",
        "thread",
    },
}

POSITIVE_CLUES = {
    "peace",
    "ceasefire",
    "agreement",
    "stable",
    "stability",
    "safe",
    "calm",
}

NEGATIVE_CLUES = {
    "war",
    "attack",
    "missile",
    "bomb",
    "dead",
    "killed",
    "threat",
    "crisis",
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
}


def remove_adjacent_duplicates(tokens: list[str]) -> list[str]:
    if not tokens:
        return tokens

    compressed = [tokens[0]]
    for token in tokens[1:]:
        if token != compressed[-1]:
            compressed.append(token)
    return compressed


def light_stem(token: str) -> str:
    # Very lightweight, dependency-free stemming to reduce feature sparsity.
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed"):
        return token[:-2]
    if len(token) > 4 and token.endswith("es"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def reduce_elongation(token: str) -> str:
    return ELONGATION_RE.sub(r"\1\1", token)


def expand_contractions(text: str) -> str:
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(contraction)}\b", expansion, text, flags=re.IGNORECASE)
    return text


def base_token(token: str) -> str:
    for prefix in ("neg_", "int_"):
        if token.startswith(prefix):
            return token[len(prefix):]
    return token


def apply_intensity_scope(tokens: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return tokens

    scoped: list[str] = []
    boost_next = False
    for token in tokens:
        root = base_token(token)
        if root in BOOSTER_WORDS:
            scoped.append(root)
            boost_next = True
            continue

        if boost_next and root not in {"exclamation", "question"}:
            scoped.append(f"int_{root}")
            boost_next = False
            continue

        scoped.append(token)

    return scoped


def apply_negation_scope(tokens: list[str], window: int) -> list[str]:
    if window <= 0:
        return tokens

    scoped: list[str] = []
    remaining_scope = 0
    for token in tokens:
        root = base_token(token)
        if root in NEGATION_WORDS:
            scoped.append(root)
            remaining_scope = window
            continue

        if root in {"exclamation", "question"}:
            scoped.append(root)
            continue

        if remaining_scope > 0:
            scoped.append(f"neg_{root}")
            remaining_scope -= 1
        else:
            scoped.append(token)

    return scoped


def inject_topic_tags(tokens: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return tokens

    token_roots = {base_token(token) for token in tokens}
    tags = [
        f"topic_{topic}"
        for topic, terms in TOPIC_TERMS.items()
        if token_roots.intersection(terms)
    ]
    return tokens + tags


def inject_sentiment_bias_tag(tokens: list[str], enabled: bool) -> list[str]:
    if not enabled:
        return tokens

    token_roots = [base_token(token) for token in tokens]
    pos_count = sum(root in POSITIVE_CLUES for root in token_roots)
    neg_count = sum(root in NEGATIVE_CLUES for root in token_roots)

    if pos_count == 0 and neg_count == 0:
        return tokens

    if pos_count > neg_count:
        return tokens + ["sentiment_pos_bias"]
    if neg_count > pos_count:
        return tokens + ["sentiment_neg_bias"]
    return tokens + ["sentiment_mixed_bias"]


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


def canonicalize_tokens(tokens: list[str]) -> list[str]:
    return [TOKEN_CANONICAL_MAP.get(token, token) for token in tokens]


def apply_corpus_frequency_filter(series: pd.Series, min_corpus_freq: int) -> pd.Series:
    if min_corpus_freq <= 1:
        return series

    token_counts: Counter[str] = Counter()
    for text in series.fillna(""):
        token_counts.update(str(text).split())

    keep_tokens = {token for token, count in token_counts.items() if count >= min_corpus_freq}
    return series.apply(
        lambda text: " ".join(token for token in str(text).split() if token in keep_tokens)
    )


def clean_text_style_b(
    text: str,
    min_token_len: int,
    keep_stopwords: bool,
    strip_non_ascii: bool,
    disable_light_stemmer: bool,
    negation_window: int,
    discard_hashtags: bool,
    disable_intensity_scope: bool,
    disable_topic_tags: bool,
    disable_sentiment_bias_tag: bool,
) -> str:
    text = "" if pd.isna(text) else str(text)
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = expand_contractions(text)

    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    if discard_hashtags:
        text = HASHTAG_RE.sub(" ", text)
    else:
        text = HASHTAG_RE.sub(lambda m: f" {m.group(1)} ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = EMOJI_RE.sub(" ", text)

    text = NUMBER_RE.sub(" ", text)
    if strip_non_ascii:
        text = text.encode("ascii", errors="ignore").decode("ascii")

    text = EXCLAIM_RE.sub(" __exclaim__ ", text)
    text = QUESTION_RE.sub(" __question__ ", text)
    punctuation = string.punctuation.replace("_", "")
    text = text.translate(str.maketrans("", "", punctuation))
    text = text.lower()
    text = WHITESPACE_RE.sub(" ", text).strip()

    if not text:
        return ""

    tokens = []
    for token in text.split(" "):
        if token == "__exclaim__":
            tokens.append("exclamation")
            continue
        if token == "__question__":
            tokens.append("question")
            continue
        tokens.append(reduce_elongation(token))

    tokens = canonicalize_tokens(tokens)
    tokens = join_domain_phrases(tokens)
    tokens = remove_adjacent_duplicates(tokens)

    if not keep_stopwords:
        tokens = [
            token for token in tokens if token not in STOPWORDS or token in NEGATION_WORDS
        ]

    if not disable_light_stemmer:
        stemmed_tokens = []
        for token in tokens:
            root = base_token(token)
            if root in {"exclamation", "question"} or "_" in root:
                stemmed_tokens.append(token)
            else:
                stemmed_tokens.append(light_stem(root))
        tokens = stemmed_tokens

    tokens = [
        token
        for token in tokens
        if base_token(token) in {"exclamation", "question"}
        or len(base_token(token)) >= min_token_len
    ]

    tokens = apply_intensity_scope(tokens, enabled=not disable_intensity_scope)
    tokens = apply_negation_scope(tokens, window=negation_window)
    tokens = inject_topic_tags(tokens, enabled=not disable_topic_tags)
    tokens = inject_sentiment_bias_tag(tokens, enabled=not disable_sentiment_bias_tag)
    tokens = remove_adjacent_duplicates(tokens)
    return " ".join(tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline B: Multi-stage lexical and contextual normalization for final_text. "
            "Removes hashtags, mentions, URLs, HTML, emojis, numbers, punctuation, "
            "normalizes whitespace, canonicalizes domain tokens/phrases, applies "
            "negation/intensity scope markers, injects topic and sentiment-bias tags, "
            "and optionally performs corpus frequency filtering."
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
        default="final_text_style_b",
        help="Name of processed text column",
    )
    parser.add_argument(
        "--min_token_len",
        type=int,
        default=2,
        help="Minimum token length to keep (default: 2)",
    )
    parser.add_argument(
        "--keep_stopwords",
        action="store_true",
        help="Keep stopwords instead of removing them",
    )
    parser.add_argument(
        "--strip_non_ascii",
        action="store_true",
        help="Drop non-ASCII characters after HTML/emoji cleanup",
    )
    parser.add_argument(
        "--disable_light_stemmer",
        action="store_true",
        help="Disable lightweight suffix-based stemming",
    )
    parser.add_argument(
        "--negation_window",
        type=int,
        default=1,
        help="How many tokens after a negator get prefixed with neg_",
    )
    parser.add_argument(
        "--discard_hashtags",
        action="store_true",
        help="Drop hashtag terms entirely instead of keeping their text",
    )
    parser.add_argument(
        "--disable_intensity_scope",
        action="store_true",
        help="Disable int_ prefixing after booster words",
    )
    parser.add_argument(
        "--disable_topic_tags",
        action="store_true",
        help="Disable topic_* tag injection",
    )
    parser.add_argument(
        "--disable_sentiment_bias_tag",
        action="store_true",
        help="Disable sentiment_*_bias tag injection",
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

    df[args.output_column] = df[args.text_column].apply(
        lambda text: clean_text_style_b(
            text=text,
            min_token_len=args.min_token_len,
            keep_stopwords=args.keep_stopwords,
            strip_non_ascii=args.strip_non_ascii,
            disable_light_stemmer=args.disable_light_stemmer,
            negation_window=args.negation_window,
            discard_hashtags=args.discard_hashtags,
            disable_intensity_scope=args.disable_intensity_scope,
            disable_topic_tags=args.disable_topic_tags,
            disable_sentiment_bias_tag=args.disable_sentiment_bias_tag,
        )
    )

    df[args.output_column] = apply_corpus_frequency_filter(
        df[args.output_column],
        min_corpus_freq=args.min_corpus_freq,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    empty_count = int((df[args.output_column].str.strip() == "").sum())
    changed_rows = int((df[args.output_column] != df[args.text_column].fillna("")).sum())
    print("Pipeline B complete.")
    print(f"Input rows: {len(df)}")
    print(f"Output saved to: {output_path}")
    print(f"Output column: {args.output_column}")
    print(f"Empty processed rows: {empty_count}")
    print(f"Rows changed vs source column '{args.text_column}': {changed_rows}")


if __name__ == "__main__":
    main()
