import pytest

import sentiment_model


def test_analyze_sentiment_uses_lazy_cached_classifier(monkeypatch):
    calls = []

    class FakeClassifier:
        def __call__(self, text):
            calls.append(text)
            return [{"label": "POSITIVE", "score": 0.97}]

    sentiment_model.get_sentiment_classifier.cache_clear()
    monkeypatch.setattr(sentiment_model, "pipeline", lambda *args, **kwargs: FakeClassifier())

    assert sentiment_model.analyze_sentiment(" I like this. ") == {
        "label": "POSITIVE",
        "score": 0.97,
    }
    assert sentiment_model.analyze_sentiment("Another example.")["label"] == "POSITIVE"
    assert calls == ["I like this.", "Another example."]


@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_analyze_sentiment_rejects_invalid_input(value):
    with pytest.raises((TypeError, ValueError)):
        sentiment_model.analyze_sentiment(value)
