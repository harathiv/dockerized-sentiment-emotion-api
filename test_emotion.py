import pytest

import emotion_model


def test_analyze_emotion_normalizes_and_sorts_scores(monkeypatch):
    class FakeClassifier:
        def __call__(self, text):
            return [[
                {"label": "sadness", "score": 0.1},
                {"label": "JOY", "score": 0.8},
                {"label": "anger", "score": 0.2},
            ]]

    emotion_model.get_emotion_classifier.cache_clear()
    monkeypatch.setattr(emotion_model, "pipeline", lambda *args, **kwargs: FakeClassifier())

    result = emotion_model.analyze_emotion("A sample", top_k=2)

    assert result["label"] == "JOY"
    assert result["score"] == 0.8
    assert result["top_emotions"] == [
        {"label": "joy", "score": 0.8},
        {"label": "anger", "score": 0.2},
    ]
    assert len(result["all_scores"]) == 3


@pytest.mark.parametrize("value", ["", "   ", None, 123])
def test_analyze_emotion_rejects_invalid_input(value):
    with pytest.raises((TypeError, ValueError)):
        emotion_model.analyze_emotion(value)
