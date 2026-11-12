from fastapi.testclient import TestClient

import api


client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_sentiment_and_emotions(monkeypatch):
    monkeypatch.setattr(
        api,
        "analyze_sentiment",
        lambda text: {"label": "POSITIVE", "score": 0.98},
    )
    monkeypatch.setattr(
        api,
        "analyze_emotion",
        lambda text, top_k: {
            "label": "JOY",
            "score": 0.91,
            "top_emotions": [{"label": "joy", "score": 0.91}],
            "all_scores": [
                {"label": "joy", "score": 0.91},
                {"label": "neutral", "score": 0.09},
            ],
        },
    )

    response = client.post("/predict", json={"text": "I am delighted."})

    assert response.status_code == 200
    assert response.json() == {
        "text": "I am delighted.",
        "sentiment": {"label": "POSITIVE", "score": 0.98},
        "emotion": {"label": "JOY", "score": 0.91},
        "top_emotions": [{"label": "joy", "score": 0.91}],
        "all_emotion_scores": [
            {"label": "joy", "score": 0.91},
            {"label": "neutral", "score": 0.09},
        ],
    }


def test_predict_rejects_blank_text():
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})
    assert response.status_code == 422
