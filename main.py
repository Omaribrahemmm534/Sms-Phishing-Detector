"""
SMS Phishing Detection — FastAPI Backend
==========================================
يستقبل رسالة نصية ويرجع هل هي phishing أم لا.

تشغيل:
    pip install fastapi uvicorn
    uvicorn main:app --reload --port 8000

API Docs (Swagger):
    http://localhost:8000/docs
"""

import os
import re
import pickle
import numpy as np
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# ─── إعداد التطبيق ─────────────────────────────────────────────────────────
app = FastAPI(
    title="SMS Phishing Detector API",
    description="كشف رسائل التصيد الاحتيالي باستخدام ML/DL",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # في الإنتاج: اكتب الـ domain بتاعك
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── نماذج Pydantic ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      example="WINNER!! You've been selected for a £1000 prize!")
    source: Optional[str] = Field(None, example="telegram")  # whatsapp/telegram/sms

class PredictResponse(BaseModel):
    label:       str    # "phishing" أو "safe"
    confidence:  float  # 0.0 → 1.0
    is_phishing: bool
    risk_level:  str    # "high" / "medium" / "low"
    features:    dict   # ميزات مساعدة للشرح
    timestamp:   str

class HealthResponse(BaseModel):
    status:     str
    model_type: str
    version:    str

# ─── تحميل النموذج ──────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "1_model", "models")
ML_MODEL   = None
DL_MODEL   = None
TOKENIZER  = None

def load_models():
    global ML_MODEL, DL_MODEL, TOKENIZER

    ml_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    if os.path.exists(ml_path):
        with open(ml_path, "rb") as f:
            ML_MODEL = pickle.load(f)
        print(f"[✓] ML model loaded from {ml_path}")
    else:
        print(f"[!] ML model not found at {ml_path} — using fallback rules")

    tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
    if os.path.exists(tok_path):
        with open(tok_path, "rb") as f:
            TOKENIZER = pickle.load(f)

    # DL model (اختياري)
    try:
        import tensorflow as tf
        import glob
        dl_files = glob.glob(os.path.join(MODELS_DIR, "best_dl_model_*.keras"))
        if dl_files:
            DL_MODEL = tf.keras.models.load_model(dl_files[0])
            print(f"[✓] DL model loaded: {dl_files[0]}")
    except Exception:
        pass

load_models()

# ─── دوال مساعدة ────────────────────────────────────────────────────────────
MAX_LEN = 150

PHISHING_KEYWORDS = [
    "winner", "prize", "congratulations", "free", "urgent", "claim",
    "verify", "account", "suspended", "click", "limited", "offer",
    "reward", "selected", "cash", "bank", "password", "confirm",
    "مبروك", "فاز", "جائزة", "عاجل", "حسابك", "تحقق", "خصم",
]

URL_PATTERN    = re.compile(r"http[s]?://\S+|www\.\S+", re.I)
PHONE_PATTERN  = re.compile(r"\b(?:\+?\d[\d\s\-]{7,}\d)\b")
CAPS_RATIO_THR = 0.4  # نسبة الأحرف الكبيرة

def preprocess(text: str) -> str:
    t = text.lower()
    t = re.sub(r"http\S+|www\S+", " url ", t)
    t = re.sub(r"\b\d+\b", " num ", t)
    t = re.sub(r"[^\w\s\u0600-\u06FF]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_features(text: str) -> dict:
    """يستخرج ميزات يدوية لتعزيز الشفافية."""
    lower = text.lower()
    return {
        "has_url":          bool(URL_PATTERN.search(text)),
        "has_phone":        bool(PHONE_PATTERN.search(text)),
        "keyword_count":    sum(1 for k in PHISHING_KEYWORDS if k in lower),
        "caps_ratio":       sum(1 for c in text if c.isupper()) / max(len(text), 1),
        "text_length":      len(text),
        "exclamation_count": text.count("!"),
        "has_numbers":       bool(re.search(r"\d+", text)),
    }

def rule_based_score(features: dict) -> float:
    """سكور بسيط يُستخدم لو النموذج مش متاح."""
    score = 0.0
    if features["has_url"]:          score += 0.3
    if features["keyword_count"] > 2: score += 0.25
    if features["caps_ratio"] > CAPS_RATIO_THR: score += 0.2
    if features["exclamation_count"] > 1: score += 0.15
    if features["has_phone"]:        score += 0.1
    return min(score, 0.99)

def get_risk_level(confidence: float, is_phishing: bool) -> str:
    if not is_phishing:
        return "low"
    if confidence >= 0.85:
        return "high"
    if confidence >= 0.6:
        return "medium"
    return "low"

def predict(text: str) -> dict:
    features = extract_features(text)
    clean    = preprocess(text)

    # 1) جرّب DL أولاً لو متاح وعنده tokenizer
    if DL_MODEL and TOKENIZER:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            seq  = TOKENIZER.texts_to_sequences([clean])
            padded = pad_sequences(seq, maxlen=MAX_LEN)
            conf = float(DL_MODEL.predict(padded, verbose=0)[0][0])
            is_p = conf >= 0.5
            return {
                "label":       "phishing" if is_p else "safe",
                "confidence":  round(conf if is_p else 1 - conf, 4),
                "is_phishing": is_p,
                "risk_level":  get_risk_level(conf, is_p),
                "features":    features,
                "model_used":  "DL (LSTM/CNN)",
            }
        except Exception:
            pass

    # 2) جرّب ML pipeline
    if ML_MODEL:
        try:
            prob = ML_MODEL.predict_proba([clean])[0][1]
            is_p = prob >= 0.5
            return {
                "label":       "phishing" if is_p else "safe",
                "confidence":  round(prob if is_p else 1 - prob, 4),
                "is_phishing": is_p,
                "risk_level":  get_risk_level(prob, is_p),
                "features":    features,
                "model_used":  "ML (TF-IDF + Classifier)",
            }
        except Exception:
            pass

    # 3) Fallback: rule-based
    score = rule_based_score(features)
    is_p  = score >= 0.5
    return {
        "label":       "phishing" if is_p else "safe",
        "confidence":  round(score if is_p else 1 - score, 4),
        "is_phishing": is_p,
        "risk_level":  get_risk_level(score, is_p),
        "features":    features,
        "model_used":  "Rule-based (fallback)",
    }

# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", tags=["General"])
def root():
    return {"message": "SMS Phishing Detector API — ارسل POST على /predict"}

@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    model_type = "DL" if DL_MODEL else ("ML" if ML_MODEL else "rule-based")
    return {"status": "ok", "model_type": model_type, "version": "1.0.0"}

@app.post("/predict", response_model=PredictResponse, tags=["Detection"])
def predict_endpoint(req: PredictRequest):
    """
    يحلل رسالة نصية ويرجع:
    - **label**: "phishing" أو "safe"
    - **confidence**: نسبة الثقة (0-1)
    - **risk_level**: high / medium / low
    - **features**: ميزات الرسالة للشفافية
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="النص فاضي!")

    result = predict(req.text)

    return PredictResponse(
        label       = result["label"],
        confidence  = result["confidence"],
        is_phishing = result["is_phishing"],
        risk_level  = result["risk_level"],
        features    = result["features"],
        timestamp   = datetime.utcnow().isoformat(),
    )

@app.post("/batch-predict", tags=["Detection"])
def batch_predict(texts: list[str]):
    """يحلل أكتر من رسالة في طلب واحد."""
    if len(texts) > 100:
        raise HTTPException(status_code=400, detail="الحد الأقصى 100 رسالة في كل طلب")
    return [predict(t) for t in texts]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
