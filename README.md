# Sentiment and Emotion Analysis Using Transformers

This project analyzes text with the existing Hugging Face transformer models in two ways:

- Sentiment classification: positive or negative
- Emotion classification: anger, disgust, fear, joy, neutral, sadness, or surprise

The original Streamlit interface remains available, and the same model modules are also exposed through a deployable FastAPI service.

## Project structure

- `app.py` — Streamlit UI
- `api.py` — FastAPI application with `GET /health` and `POST /predict`
- `sentiment_model.py` — lazily loaded sentiment model
- `emotion_model.py` — lazily loaded seven-class emotion model
- `evaluation.py` — CSV evaluation script
- `test_*.py` — Pytest unit and API tests
- `Dockerfile` — container image for the API
- `.github/workflows/ci.yml` — test and Docker build workflow

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Models are downloaded by Hugging Face the first time a prediction is made. Importing the API and running unit tests does not download models.

## Run the Streamlit app

```powershell
streamlit run app.py
```

## Run the API

```powershell
uvicorn api:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

Prediction:

```powershell
curl -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"text\":\"I am thrilled to start this new project.\"}'
```

The prediction response includes the original text, sentiment label and confidence, the top emotion and confidence, the top emotions, and the complete emotion score distribution.

## Run with Docker

```powershell
docker build -t sentiment-emotion-api .
docker run --rm -p 8000:8000 sentiment-emotion-api
```

Then call `http://127.0.0.1:8000/health` or `POST /predict` as shown above.

## Testing and evaluation

Run the automated tests:

```powershell
pytest -q
```

Run the optional CSV evaluation:

```powershell
python evaluation.py
```

GitHub Actions runs Pytest and builds the Docker image on pushes and pull requests.

## Example

Input:

> I am thrilled to start this new project.

Typical output:

- Sentiment: `POSITIVE`
- Emotion: `JOY`

The models are pretrained and may still misclassify sarcasm, slang, or subtle emotional language.
