"""
SMS Phishing Detection — Streamlit Dashboard
=============================================
واجهة مستخدم تفاعلية للتحليل وعرض الإحصائيات.

تشغيل:
    pip install streamlit requests pandas plotly
    streamlit run app.py
"""

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime

# ─── إعداد الصفحة ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SMS Phishing Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

# ─── Session State ────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # قائمة من dicts

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/spam.png", width=60)
    st.title("🛡️ Phishing Detector")
    st.caption("مشروع تخرج — SMS Phishing Detection using ML & DL")
    st.divider()
    st.subheader("⚙️ الإعدادات")
    api_url = st.text_input("API URL", value=API_URL)
    threshold = st.slider("حد الكشف (Threshold)", 0.3, 0.9, 0.5, 0.05)
    st.divider()

    # صحة الـ API
    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        if r.status_code == 200:
            h = r.json()
            st.success(f"✅ API شغّال\nنموذج: {h['model_type']}")
        else:
            st.error("❌ API مش شغّال")
    except Exception:
        st.warning("⚠️ تعذّر الاتصال بالـ API\nتأكد من تشغيل `uvicorn main:app`")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🔍 كشف رسائل التصيد الاحتيالي")
st.caption("SMS Phishing Detection using Machine Learning & Deep Learning")
st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_stats, tab_about = st.tabs([
    "🔎 تحليل رسالة",
    "📦 تحليل مجموعة",
    "📊 الإحصائيات",
    "ℹ️ عن المشروع",
])

# ══════════════════════════════════════════════════════════════════════════════
# Tab 1: تحليل رسالة واحدة
# ══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📩 أدخل الرسالة")
        sample = st.selectbox("جرب رسالة نموذجية:", [
            "اكتب رسالتك هنا...",
            "WINNER!! You've been selected for a £1000 prize. Call now!",
            "Hey, are you coming to the party tonight?",
            "URGENT: Your account has been suspended. Verify immediately",
            "مبروك! لقد فزت بجائزة نقدية. اتصل الآن لاستلامها",
            "هل أنت جاهز للاجتماع غدًا الساعة 3؟",
        ])

        text_input = st.text_area(
            "نص الرسالة:",
            value=sample if sample != "اكتب رسالتك هنا..." else "",
            height=140,
            placeholder="الصق الرسالة المشبوهة هنا...",
        )
        source = st.selectbox("مصدر الرسالة:", ["sms", "whatsapp", "telegram", "email", "other"])

        analyze_btn = st.button("🔍 تحليل", type="primary", use_container_width=True)

    with col2:
        st.subheader("📋 النتيجة")

        if analyze_btn and text_input.strip():
            with st.spinner("جارٍ التحليل..."):
                try:
                    resp = requests.post(
                        f"{api_url}/predict",
                        json={"text": text_input, "source": source},
                        timeout=10,
                    )
                    result = resp.json()
                    conf = result["confidence"]
                    is_p = conf >= threshold if result["is_phishing"] else False

                    # عرض النتيجة
                    if result["is_phishing"]:
                        st.error(f"🔴 **احتيالية** — ثقة: {conf*100:.1f}%")
                    else:
                        st.success(f"🟢 **آمنة** — ثقة: {conf*100:.1f}%")

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=conf * 100,
                        title={"text": "نسبة الخطر %"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar":  {"color": "#e74c3c" if result["is_phishing"] else "#27ae60"},
                            "steps": [
                                {"range": [0, 40],   "color": "#d5f5e3"},
                                {"range": [40, 70],  "color": "#fef9e7"},
                                {"range": [70, 100], "color": "#fadbd8"},
                            ],
                            "threshold": {"line": {"color": "black", "width": 3}, "value": threshold * 100},
                        },
                        number={"suffix": "%"},
                    ))
                    fig.update_layout(height=220, margin=dict(t=30, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)

                    # الميزات
                    feats = result.get("features", {})
                    st.markdown("**🔬 ميزات الرسالة:**")
                    feat_cols = st.columns(2)
                    items = [
                        ("🔗 يحتوي URL",      "✅" if feats.get("has_url") else "❌"),
                        ("📞 يحتوي رقم",      "✅" if feats.get("has_phone") else "❌"),
                        ("🚨 كلمات مشبوهة",  str(feats.get("keyword_count", 0))),
                        ("🔠 نسبة Caps",      f"{feats.get('caps_ratio', 0)*100:.1f}%"),
                        ("📏 طول الرسالة",   str(feats.get("text_length", 0))),
                        ("❗ علامات تعجب",   str(feats.get("exclamation_count", 0))),
                    ]
                    for i, (k, v) in enumerate(items):
                        feat_cols[i % 2].metric(k, v)

                    # حفظ في التاريخ
                    st.session_state.history.append({
                        "text":       text_input[:80] + "..." if len(text_input) > 80 else text_input,
                        "label":      result["label"],
                        "confidence": conf,
                        "risk":       result["risk_level"],
                        "source":     source,
                        "time":       datetime.now().strftime("%H:%M:%S"),
                    })

                except requests.ConnectionError:
                    st.error("❌ تعذر الاتصال بالـ API!")
                except Exception as e:
                    st.error(f"خطأ: {e}")

        elif analyze_btn:
            st.warning("⚠️ اكتب رسالة أولاً")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 2: Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("📦 تحليل مجموعة رسائل")
    batch_input = st.text_area(
        "أدخل الرسائل (رسالة في كل سطر):",
        height=200,
        placeholder="الرسالة الأولى\nالرسالة الثانية\n..."
    )
    if st.button("🚀 تحليل المجموعة", type="primary"):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if not lines:
            st.warning("أدخل رسائل أولاً")
        else:
            with st.spinner(f"جارٍ تحليل {len(lines)} رسالة..."):
                try:
                    resp = requests.post(f"{api_url}/batch-predict", json=lines, timeout=30)
                    results = resp.json()
                    rows = []
                    for msg, r in zip(lines, results):
                        rows.append({
                            "الرسالة":    msg[:60] + "..." if len(msg) > 60 else msg,
                            "التصنيف":    "🔴 احتيالية" if r["is_phishing"] else "🟢 آمنة",
                            "الثقة %":   f"{r['confidence']*100:.1f}",
                            "مستوى الخطر": r["risk_level"],
                        })
                    df = pd.DataFrame(rows)
                    st.dataframe(df, use_container_width=True)

                    phish_count = sum(1 for r in results if r["is_phishing"])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("إجمالي الرسائل", len(results))
                    c2.metric("احتيالية", phish_count, delta_color="inverse")
                    c3.metric("آمنة", len(results) - phish_count)
                except Exception as e:
                    st.error(f"خطأ: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Tab 3: Statistics
# ══════════════════════════════════════════════════════════════════════════════
with tab_stats:
    st.subheader("📊 إحصائيات الجلسة")

    if not st.session_state.history:
        st.info("لسه ما حللتش أي رسائل في الجلسة دي.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        total   = len(hist_df)
        phish   = (hist_df["label"] == "phishing").sum()
        safe    = total - phish

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📩 إجمالي", total)
        c2.metric("🔴 احتيالية", phish)
        c3.metric("🟢 آمنة", safe)
        c4.metric("📈 نسبة الكشف", f"{phish/total*100:.1f}%")

        col_a, col_b = st.columns(2)
        with col_a:
            fig_pie = px.pie(
                hist_df, names="label",
                color="label",
                color_discrete_map={"phishing": "#e74c3c", "safe": "#27ae60"},
                title="توزيع الرسائل",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            fig_hist = px.histogram(
                hist_df, x="confidence", color="label",
                color_discrete_map={"phishing": "#e74c3c", "safe": "#27ae60"},
                title="توزيع نسب الثقة",
                nbins=20,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.subheader("📋 سجل التحليلات")
        st.dataframe(hist_df, use_container_width=True)

        if st.button("🗑️ مسح السجل"):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# Tab 4: About
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ عن المشروع")
    st.markdown("""
    ### SMS Phishing Detection using ML & DL

    **الهدف:** بناء نظام ذكي لكشف رسائل التصيد الاحتيالي (Phishing/Spam)
    في SMS و WhatsApp و Telegram.

    ---

    #### 🏗️ المكونات:
    | المكون | التقنية |
    |--------|---------|
    | النماذج | Naive Bayes, SVM, LSTM, CNN-Text |
    | الـ Features | TF-IDF, Word Embeddings, Rule-based |
    | الـ Backend | FastAPI (Python) |
    | الـ Dashboard | Streamlit |
    | تكامل المنصات | Telegram Bot API, Twilio (WhatsApp) |

    #### 📊 مقاييس التقييم:
    - **Accuracy** — الدقة الكلية
    - **Precision** — دقة الكشف عن الـ Spam
    - **Recall** — نسبة اكتشاف الـ Spam الفعلي
    - **F1-Score** — التوازن بين Precision و Recall
    - **AUC-ROC** — جودة النموذج الكلية

    ---
    *مشروع تخرج — كلية الحاسبات*
    """)
