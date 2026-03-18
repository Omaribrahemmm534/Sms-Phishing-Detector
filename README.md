# 🛡️ SMS Phishing Detection

مشروع تخرج لكشف رسائل التصيد الاحتيالي باستخدام Machine Learning و Deep Learning،
مع تكامل مع Telegram و WhatsApp.

---

## 📁 هيكل المشروع

```
phishing-detector/
├── 1_model/
│   ├── train.py          ← تدريب النماذج (ML + DL)
│   └── models/           ← النماذج المحفوظة (بيتعمل تلقائياً)
├── 2_api/
│   └── main.py           ← FastAPI Backend
├── 3_bots/
│   ├── telegram_bot.py   ← بوت تيليجرام
│   └── whatsapp_bot.py   ← تكامل واتساب (Twilio)
├── 4_dashboard/
│   └── app.py            ← Streamlit Dashboard
└── requirements.txt
```

---

## ⚙️ التثبيت والتشغيل

### 1. تثبيت المكتبات
```bash
pip install -r requirements.txt
```

### 2. تحميل الـ Dataset
حمّل ملف `SMSSpamCollection` من:
https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

ضعه في مجلد `1_model/`

### 3. تدريب النموذج
```bash
cd 1_model
python train.py
```
بعد التدريب، هيتعمل مجلد `models/` فيه النماذج المحفوظة.

### 4. تشغيل الـ API
```bash
cd 2_api
uvicorn main:app --reload --port 8000
```
افتح: http://localhost:8000/docs

### 5. تشغيل الـ Dashboard
```bash
cd 4_dashboard
streamlit run app.py
```

### 6. تشغيل بوت تيليجرام
```bash
# أولاً: احصل على token من @BotFather
export TELEGRAM_TOKEN="your_token_here"
cd 3_bots
python telegram_bot.py
```

---

## 🧠 النماذج المستخدمة

| النموذج | النوع | الميزة |
|---------|-------|--------|
| Naive Bayes | ML | سريع ومناسب للنص |
| SVM (LinearSVC) | ML | دقيق للنصوص القصيرة |
| Logistic Regression | ML | سهل التفسير |
| Random Forest | ML | مقاوم للـ Overfitting |
| LSTM | DL | يفهم السياق |
| CNN-Text | DL | أسرع من LSTM |

---

## 📡 API Endpoints

| Method | Endpoint | الوصف |
|--------|----------|-------|
| GET | `/health` | حالة الـ API |
| POST | `/predict` | تحليل رسالة واحدة |
| POST | `/batch-predict` | تحليل مجموعة رسائل |

### مثال على الاستخدام:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER!! You have been selected for a prize!"}'
```

### الرد:
```json
{
  "label": "phishing",
  "confidence": 0.97,
  "is_phishing": true,
  "risk_level": "high",
  "features": {
    "has_url": false,
    "has_phone": false,
    "keyword_count": 3,
    "caps_ratio": 0.15
  }
}
```

---

## 📊 مقاييس التقييم المتوقعة

على UCI SMS Spam Collection:
- **Accuracy**: ~98%
- **F1-Score**: ~97%
- **AUC-ROC**: ~99%

---

*مشروع تخرج — كلية الحاسبات*
