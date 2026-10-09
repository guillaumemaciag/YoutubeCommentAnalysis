"""Sentiment scoring and lightweight text summaries."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import re

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "before",
    "being", "but", "can", "could", "did", "does", "doing", "don", "for", "from",
    "get", "got", "had", "has", "have", "here", "how", "into", "its", "just",
    "like", "more", "most", "not", "now", "one", "only", "out", "really", "should",
    "some", "than", "that", "the", "their", "them", "then", "there", "these", "they",
    "thing", "this", "those", "too", "very", "video", "was", "way", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "would", "you", "your",
    "youtube", "https", "www", "com",
}


def sentiment_label(compound: float) -> str:
    """Apply VADER's documented compound-score thresholds."""
    if compound >= 0.05:
        return "Positive"
    if compound <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_comments(comments: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Return comment records enriched with VADER sentiment scores."""
    records = list(comments)
    base_columns = [
        "comment_id", "parent_id", "is_reply", "author", "text", "published_at",
        "updated_at", "likes", "reply_count",
    ]
    output_columns = base_columns + ["positive", "neutral", "negative", "compound", "sentiment"]
    if not records:
        return pd.DataFrame(columns=output_columns)

    analyzer = SentimentIntensityAnalyzer()
    enriched: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        scores = analyzer.polarity_scores(str(row.get("text", "")))
        row.update(
            {
                "positive": scores["pos"],
                "neutral": scores["neu"],
                "negative": scores["neg"],
                "compound": scores["compound"],
                "sentiment": sentiment_label(scores["compound"]),
            }
        )
        enriched.append(row)

    frame = pd.DataFrame(enriched)
    for column in output_columns:
        if column not in frame:
            frame[column] = None
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame["updated_at"] = pd.to_datetime(frame["updated_at"], utc=True, errors="coerce")
    return frame[output_columns]


def top_keywords(texts: Iterable[str], limit: int = 15) -> pd.DataFrame:
    """Count meaningful words in a sequence of comments."""
    counts: Counter[str] = Counter()
    for text in texts:
        words = (word.lower().strip("'-") for word in TOKEN_PATTERN.findall(str(text)))
        counts.update(word for word in words if word and word not in STOP_WORDS)
    return pd.DataFrame(counts.most_common(limit), columns=["keyword", "mentions"])
