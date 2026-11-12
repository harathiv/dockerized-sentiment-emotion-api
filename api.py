from typing import Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, field_validator

from emotion_model import analyze_emotion
from sentiment_model import analyze_sentiment

app = FastAPI(
    title="Sentiment and Emotion Analysis API",
    description="Classify text with the project's existing transformer models.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value


class Score(BaseModel):
    label: str
    score: float


class PredictionResponse(BaseModel):
    text: str
    sentiment: Score
    emotion: Score
    top_emotions: List[Score]
    all_emotion_scores: List[Score]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    sentiment = analyze_sentiment(request.text)
    emotion = analyze_emotion(request.text, top_k=5)

    return PredictionResponse(
        text=request.text,
        sentiment=Score(**sentiment),
        emotion=Score(label=emotion["label"], score=emotion["score"]),
        top_emotions=[Score(**item) for item in emotion["top_emotions"]],
        all_emotion_scores=[Score(**item) for item in emotion["all_scores"]],
    )
