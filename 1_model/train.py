"""
SMS Phishing Detection — Model Training
========================================
يدرّب عدة نماذج ML و DL ويحفظ أفضلهم.
يستخدم dataset من UCI SMS Spam Collection.

تشغيل:
    pip install -r requirements.txt
    python train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import re
import warnings
warnings.filterwarnings("ignore")

# ─── TensorFlow / Keras (DL) ───────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Embedding, LSTM, Dense, Dropout,
        GlobalMaxPooling1D, Conv1D
    )
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False
    print("[!] TensorFlow مش متاح — هيتدرب ML فقط")

# ─── Config ────────────────────────────────────────────────────────────────
MAX_FEATURES   = 10_000   # حجم المفردات للـ TF-IDF والـ Tokenizer
MAX_LEN        = 150      # أقصى طول للرسالة (في الـ DL)
EMBEDDING_DIM  = 64       # حجم الـ embedding layer
LSTM_UNITS     = 64
EPOCHS         = 10
BATCH_SIZE     = 32
TEST_SIZE      = 0.2
RANDOM_STATE   = 42
MODELS_DIR     = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. تحميل وتنظيف البيانات
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str = None) -> pd.DataFrame:
    """
    يحمّل dataset من مسار محدد، أو يولّد بيانات تجريبية لو مفيش ملف.

    للاستخدام الحقيقي:
        حمّل الملف من: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
        ضعه في نفس المجلد باسم: SMSSpamCollection
    """
    if path and os.path.exists(path):
        df = pd.read_csv(path, encoding="latin-1")[["v1", "v2"]].rename(
         columns={"v1": "label", "v2": "text"}
)
        df["label"] = df["label"].map({"ham": 0, "spam": 1})    
        print(f"[+] تم تحميل {len(df)} رسالة من {path}")
    else:
        print("[!] Dataset مش موجود — بييولّد بيانات تجريبية (demo only)")
        df = _generate_demo_data()

    return df


def _generate_demo_data(n: int = 2000) -> pd.DataFrame:
    """يولّد بيانات تجريبية لاختبار الكود بدون dataset حقيقي."""
    spam_templates = [
        "WINNER!! You have been selected to receive a £1000 prize. Call now!",
        "FREE entry in 2 a wkly comp to win FA Cup. Text FA to 87121",
        "Congratulations! You've won a free iPhone. Click here to claim",
        "URGENT: Your account has been compromised. Verify immediately",
        "You have won $5000 cash prize. Send your details to collect",
        "Limited time offer! Get 90% off. Reply YES to claim your discount",
        "Your bank account will be suspended. Update your info now",
        "Click the link to verify your identity or your account will be closed",
        "Free ringtones! Reply WIN to 8007 to get yours now",
        "You are selected for a cash reward. Call 0800 to claim now",
        "مبروك! لقد فزت بجائزة مالية. اتصل الآن لاستلام جائزتك",
        "تحذير عاجل: حسابك البنكي مهدد. تحقق من بياناتك فورًا",
        "عرض خاص لك فقط! احصل على خصم 80% على جميع منتجاتنا",
    ]
    ham_templates = [
        "Hey, are you coming to the party tonight?",
        "Can you pick up some milk on the way home?",
        "Meeting at 3pm tomorrow, don't forget!",
        "Happy birthday! Hope you have a great day",
        "Just finished the report, sending it now",
        "Are you free this weekend for a coffee?",
        "I'll be there in 10 minutes",
        "Thanks for your help yesterday, really appreciated",
        "The match starts at 8, want to watch together?",
        "Call me when you get a chance",
        "مرحبًا، كيف حالك؟",
        "هل أنت جاهز للاجتماع غدًا؟",
        "شكرًا على مساعدتك، كنت رائعًا",
    ]
    rng = np.random.default_rng(RANDOM_STATE)
    spam_msgs = [rng.choice(spam_templates) for _ in range(n // 4)]
    ham_msgs  = [rng.choice(ham_templates)  for _ in range(n - n // 4)]
    labels    = [1] * len(spam_msgs) + [0] * len(ham_msgs)
    texts     = spam_msgs + ham_msgs
    df = pd.DataFrame({"label": labels, "text": texts})
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def preprocess_text(text: str) -> str:
    """تنظيف النص: إزالة URLs، أرقام، وعلامات الترقيم الزايدة."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " url ", text)          # URLs
    text = re.sub(r"\b\d+\b", " num ", text)                  # أرقام
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)        # ترقيم (مع دعم عربي)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ══════════════════════════════════════════════════════════════════════════════
# 2. بناء وتدريب نماذج ML
# ══════════════════════════════════════════════════════════════════════════════

def build_ml_pipelines() -> dict:
    """يبني pipelines جاهزة: TF-IDF + classifier."""
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=(1, 2),        # unigrams + bigrams
        sublinear_tf=True,
    )
    return {
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2))),
            ("clf",   MultinomialNB(alpha=0.1)),
        ]),
        "SVM (LinearSVC)": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   LinearSVC(C=1.0, max_iter=2000)),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 2), sublinear_tf=True)),
            ("clf",   LogisticRegression(C=1.0, max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=MAX_FEATURES)),
            ("clf",   RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ]),
    }


def train_ml_models(X_train, X_test, y_train, y_test) -> dict:
    """يدرّب ويقيّم كل نماذج ML."""
    pipelines = build_ml_pipelines()
    results   = {}

    print("\n" + "="*50)
    print("  نتائج نماذج Machine Learning")
    print("="*50)

    best_model = None
    best_f1    = 0.0

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        try:
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc    = roc_auc_score(y_test, y_prob)
        except AttributeError:
            auc = None  # LinearSVC لا يدعم predict_proba

        report = classification_report(y_test, y_pred, output_dict=True)
        acc    = accuracy_score(y_test, y_pred)
        f1     = report["1"]["f1-score"]

        results[name] = {
            "accuracy": acc,
            "precision": report["1"]["precision"],
            "recall":    report["1"]["recall"],
            "f1":        f1,
            "auc":       auc,
            "model":     pipeline,
        }

        print(f"\n[ {name} ]")
        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {report['1']['precision']:.4f}")
        print(f"  Recall    : {report['1']['recall']:.4f}")
        print(f"  F1-Score  : {f1:.4f}")
        if auc:
            print(f"  AUC-ROC   : {auc:.4f}")

        if f1 > best_f1:
            best_f1    = f1
            best_model = (name, pipeline)

    print(f"\n[★] أفضل نموذج ML: {best_model[0]} (F1={best_f1:.4f})")
    return results, best_model


# ══════════════════════════════════════════════════════════════════════════════
# 3. بناء وتدريب نماذج DL
# ══════════════════════════════════════════════════════════════════════════════

def prepare_dl_data(X_train, X_test):
    """يحوّل النصوص لـ sequences للـ LSTM."""
    tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_tr = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
    X_te = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN)

    return X_tr, X_te, tokenizer


def build_lstm_model() -> "tf.keras.Model":
    """نموذج LSTM أساسي."""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        LSTM(LSTM_UNITS, return_sequences=True),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_cnn_model() -> "tf.keras.Model":
    """نموذج CNN-Text أسرع من LSTM."""
    model = Sequential([
        Embedding(MAX_FEATURES, EMBEDDING_DIM, input_length=MAX_LEN),
        Conv1D(128, 5, activation="relu"),
        GlobalMaxPooling1D(),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_dl_models(X_train_seq, X_test_seq, y_train, y_test) -> dict:
    """يدرّب نماذج Deep Learning."""
    dl_models = {
        "LSTM":     build_lstm_model(),
        "CNN-Text": build_cnn_model(),
    }
    results = {}
    early_stop = EarlyStopping(patience=2, restore_best_weights=True)

    print("\n" + "="*50)
    print("  نتائج نماذج Deep Learning")
    print("="*50)

    for name, model in dl_models.items():
        print(f"\n[ تدريب {name} ]")
        model.fit(
            X_train_seq, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=1,
        )
        y_prob = model.predict(X_test_seq, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        report = classification_report(y_test, y_pred, output_dict=True)
        acc    = accuracy_score(y_test, y_pred)
        auc    = roc_auc_score(y_test, y_prob)

        results[name] = {
            "accuracy": acc,
            "f1":       report["1"]["f1-score"],
            "auc":      auc,
            "model":    model,
        }

        print(f"  Accuracy : {acc:.4f}  |  F1: {report['1']['f1-score']:.4f}  |  AUC: {auc:.4f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. حفظ النماذج
# ══════════════════════════════════════════════════════════════════════════════

def save_models(best_ml, tokenizer=None, best_dl_model=None, best_dl_name=None):
    """يحفظ الـ pipeline الأفضل والـ tokenizer."""
    name, pipeline = best_ml

    # حفظ ML pipeline (يشمل TF-IDF + classifier)
    ml_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    with open(ml_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"\n[✓] تم حفظ نموذج ML في: {ml_path}  ({name})")

    # حفظ Tokenizer
    if tokenizer:
        tok_path = os.path.join(MODELS_DIR, "tokenizer.pkl")
        with open(tok_path, "wb") as f:
            pickle.dump(tokenizer, f)
        print(f"[✓] تم حفظ Tokenizer في: {tok_path}")

    # حفظ DL model
    if best_dl_model:
        dl_path = os.path.join(MODELS_DIR, f"best_dl_model_{best_dl_name}.keras")
        best_dl_model.save(dl_path)
        print(f"[✓] تم حفظ نموذج DL في: {dl_path}  ({best_dl_name})")


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════╗")
    print("║  SMS Phishing Detection — Model Training ║")
    print("╚══════════════════════════════════════════╝\n")

    # --- تحميل البيانات ---
    df = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})  
    df["text"] = df["text"].apply(preprocess_text)

    print(f"\n[i] توزيع البيانات:")
    print(df["label"].value_counts().rename({0: "Ham (Safe)", 1: "Spam (Phishing)"}).to_string())

    X = df["text"].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # --- ML ---
    ml_results, best_ml = train_ml_models(X_train, X_test, y_train, y_test)

    # --- DL ---
    tokenizer = None
    best_dl_model, best_dl_name = None, None

    if DL_AVAILABLE:
        X_tr_seq, X_te_seq, tokenizer = prepare_dl_data(X_train, X_test)
        dl_results = train_dl_models(X_tr_seq, X_te_seq, y_train, y_test)

        # نختار أفضل DL model
        best_dl_name  = max(dl_results, key=lambda k: dl_results[k]["f1"])
        best_dl_model = dl_results[best_dl_name]["model"]

        # مقارنة النتائج الكلية
        print("\n" + "="*50)
        print("  المقارنة الكلية")
        print("="*50)
        all_results = {**{k: v for k, v in ml_results.items()},
                       **{k: v for k, v in dl_results.items()}}
        comparison = pd.DataFrame([
            {"Model": k, "Accuracy": v["accuracy"], "F1": v["f1"],
             "AUC": v.get("auc", None)}
            for k, v in all_results.items()
        ]).sort_values("F1", ascending=False)
        print(comparison.to_string(index=False))

    # --- حفظ ---
    save_models(best_ml, tokenizer, best_dl_model, best_dl_name)
    print("\n[✓] انتهى التدريب بنجاح!")


if __name__ == "__main__":
    main()
