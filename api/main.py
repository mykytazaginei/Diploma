# api/main.py

import os
import re
import joblib
import spacy
import math
from fastapi import FastAPI, HTTPException  # type: ignore
from pydantic import BaseModel
import sklearn
import gradio as gr   # type: ignore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.join(BASE_DIR, "..", "saved_pipeline_spacy_svm")
PIPELINE_PATH = os.path.join(PIPELINE_DIR, "best_toxicity_pipeline.joblib")

pipeline = None
nlp = None

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    input_text: str
    is_toxic: bool
    confidence_score: float

def preprocess_text_spacy(text: str) -> str:
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if nlp:
        doc = nlp(text)
        processed_tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
        return ' '.join(processed_tokens)
    else:
        return text

app = FastAPI(
    title="Toxicity Detector API & UI",
    description="API and simple UI for the toxicity detector.",
    version="1.2.1" 
)

@app.on_event("startup")
async def load_resources():
    """Load spaCy and the best pipeline on API startup."""
    global pipeline, nlp
    print("Loading API resources (pipeline)...")
    try:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        print(f"spaCy model 'en_core_web_sm' loaded (spaCy version: {spacy.__version__})")

        if os.path.exists(PIPELINE_PATH):
            pipeline = joblib.load(PIPELINE_PATH)
            print(f"Best pipeline loaded from {PIPELINE_PATH}")
            if hasattr(pipeline, 'steps') and hasattr(pipeline.steps[-1][1], '_sklearn_version'):
                 model_sklearn_version = pipeline.steps[-1][1]._sklearn_version
                 print(f"Pipeline sklearn version: {model_sklearn_version}")
            else:
                 print("Pipeline sklearn version not found.")
            print(f"Current scikit-learn version: {sklearn.__version__}")
        else:
            print(f"[ERROR] Pipeline file not found: {PIPELINE_PATH}")

    except Exception as e:
        print(f"[CRITICAL ERROR] Could not load resources: {e}")

    if not pipeline or not nlp:
        print("[WARNING] Not all resources loaded! API or UI might not work.")
    else:
        print("All resources loaded successfully.")


def predict_gradio(input_text: str) -> tuple[str, float]:
    """
    Function called by the Gradio interface.
    Takes text input, returns (label, confidence) tuple.
    """
    if not pipeline or not nlp:
        return "Error: Model not loaded", 0.0
    if not input_text or not input_text.strip():
        return None, 0.0

    try:
        processed_text = preprocess_text_spacy(input_text)
        score = pipeline.decision_function([processed_text])[0]
        is_toxic = bool(score > 0)
        confidence = 1 / (1 + math.exp(-score))
        label = "Toxic" if is_toxic else "Non-Toxic"
        return label, confidence
    except Exception as e:
        print(f"Error in Gradio predict for '{input_text[:50]}...': {e}")
        return f"Processing Error: {e}", 0.0

input_component = gr.Textbox(lines=5, label="Comment", placeholder="Enter comment for analysis...")
output_label = gr.Label(label="Result")
output_confidence = gr.Number(label="Confidence Score (0=Non-Toxic, 1=Toxic)")

gradio_interface = gr.Interface(
    fn=predict_gradio,
    inputs=input_component,
    outputs=[output_label, output_confidence],
    title="Comment Toxicity Detector", 
    description="Enter a comment to determine if it is toxic. Model: spaCy + TF-IDF + LinearSVC (tuned).", 
    allow_flagging="never"
)

app = gr.mount_gradio_app(
    app,
    gradio_interface,
    path="/ui"
)

@app.get("/")
async def read_root():
    """Check if API is running."""
    return {
        "status": "ok",
        "message": "Toxicity Detector API (Optimized) is running.",
        "docs_url": "/docs",
        "ui_url": "/ui"
        }

@app.post("/predict/", response_model=PredictionOutput)
async def predict_toxicity(payload: TextInput):
    """Predict toxicity for the input text (API endpoint)."""
    if not pipeline or not nlp:
        raise HTTPException(status_code=503, detail="Model resources (pipeline) not loaded.")
    if not payload.text or not payload.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' cannot be empty.")

    input_text = payload.text
    try:
        processed_text = preprocess_text_spacy(input_text)
        score = pipeline.decision_function([processed_text])[0]
        is_toxic = bool(score > 0)
        confidence = 1 / (1 + math.exp(-score)) # Sigmoid

        return PredictionOutput(
            input_text=input_text,
            is_toxic=is_toxic,
            confidence_score=confidence
        )
    except Exception as e:
        print(f"Error during prediction for text '{input_text[:50]}...': {e}")
        raise HTTPException(status_code=500, detail=f"Server error during processing: {e}")
    
#source .venv/bin/activate  
#deactivate   
#http://127.0.0.1:8000/ui
#uvicorn api.main:app --reload --host 0.0.0.0 --port 8000