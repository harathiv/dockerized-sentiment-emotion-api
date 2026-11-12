from functools import lru_cache

from transformers import pipeline

SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

@lru_cache(maxsize=1)
def get_sentiment_classifier():
    """Load the sentiment pipeline once, only when it is first needed."""
    return pipeline("sentiment-analysis", model=SENTIMENT_MODEL)


def analyze_sentiment(text):
    """Return the sentiment label and confidence score for valid text."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()

    if not text:
        raise ValueError("Input text cannot be empty.")

    result = get_sentiment_classifier()(text)[0]

    return {
        "label": result["label"],
        "score": result["score"],
    }