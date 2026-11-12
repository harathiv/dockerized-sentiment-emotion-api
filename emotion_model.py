from functools import lru_cache

from transformers import pipeline

EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

@lru_cache(maxsize=1)
def get_emotion_classifier():
    """Load the emotion pipeline once, only when it is first needed."""
    return pipeline("text-classification", model=EMOTION_MODEL, top_k=None)


def analyze_emotion(text, top_k=3):
    """Return the top predicted emotion and the score distribution."""
    if not isinstance(text, str):
        raise TypeError("Input must be a string.")

    text = text.strip()
    if not text:
        raise ValueError("Input text cannot be empty.")

    output = get_emotion_classifier()(text)

    if isinstance(output, list) and output and isinstance(output[0], list):
        scores = output[0]
    elif isinstance(output, list) and output and isinstance(output[0], dict):
        scores = output
    else:
        raise ValueError("Unexpected emotion model output format.")

    normalized = [
        {
            "label": str(item["label"]).lower(),
            "score": float(item["score"]),
        }
        for item in scores
    ]
    normalized.sort(key=lambda item: item["score"], reverse=True)

    top_emotions = normalized[:max(1, int(top_k))]
    top_emotion = top_emotions[0]

    return {
        "label": top_emotion["label"].upper(),
        "score": top_emotion["score"],
        "top_emotions": top_emotions,
        "all_scores": normalized,
    }
