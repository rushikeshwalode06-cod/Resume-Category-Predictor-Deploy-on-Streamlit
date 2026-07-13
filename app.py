
import streamlit as st
import pickle

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("resume.pkl", "rb"))
label_encoder = pickle.load(open("label_en.pkl", "rb"))

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Resume Category Predictor",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

.stApp{
    background-color:#f5f7fa;
}

.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:white;
}

.subtitle{
    text-align:center;
    color:white;
    font-size:18px;
}

.header{
    background: linear-gradient(90deg,#0f4c81,#00b4d8);
    padding:25px;
    border-radius:15px;
}

.result{
    background:#d4edda;
    padding:20px;
    border-radius:12px;
    text-align:center;
    font-size:28px;
    color:#155724;
    font-weight:bold;
}

.footer{
    text-align:center;
    color:gray;
    font-size:15px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
<div class="title">📄 Resume Category Predictor</div>
<div class="subtitle">
Predict Resume Category using Machine Learning (TF-IDF + Pipeline)
</div>
</div>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("📌 Project Details")

st.sidebar.success("Machine Learning Project")

st.sidebar.write("""
### Algorithm
✅ TF-IDF

✅ Pipeline

✅ Machine Learning

### Model Files
- resume.pkl
- label_en.pkl

### Developed By
Rushikesh Walode
""")

# -----------------------------
# Resume Input
# -----------------------------
resume = st.text_area(
    "📄 Paste Resume Here",
    height=300,
    placeholder="Paste your complete resume..."
)

st.write("")

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Category", use_container_width=True):

    if resume.strip() == "":
        st.warning("⚠ Please paste your resume first.")
    else:

        prediction = model.predict([resume])

        category = label_encoder.inverse_transform(prediction)

        st.balloons()

        st.markdown(
            f"""
            <div class="result">
            🎯 Predicted Category <br><br>
            {category[0]}
            </div>
            """,
            unsafe_allow_html=True
        )

# -----------------------------
# Footer
# -----------------------------
st.write("")
st.write("---")

st.markdown("""
<div class="footer">
Developed with ❤️ using Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)