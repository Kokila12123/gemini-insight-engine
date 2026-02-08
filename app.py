import streamlit as st
import pandas as pd
import google.generativeai as genai

# ================== CONFIG ==================
MODEL_NAME = "models/gemini-3-flash-preview"

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(
    page_title="Gemini Insight Engine",
    page_icon="🔍",
    layout="wide"
)

# ================== FUNCTIONS ==================
def build_prompt(dataframe: pd.DataFrame) -> str:
    return f"""
You are a senior data analyst.

Analyze the following dataset:

{dataframe.head(50).to_csv(index=False)}

Provide:
1. Key trends
2. Anomalies or unusual behavior
3. Risks or concerns
4. Clear, actionable recommendations

Explain everything in simple, non-technical language.
"""


def analyze_with_gemini(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# ================== SIDEBAR ==================
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload a dataset and generate AI-powered insights.")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

run_analysis = st.sidebar.button("🔍 Analyze Data")

# ================== MAIN UI ==================
st.title("🔍 Gemini Insight Engine")
st.caption("Turn raw data into clear, actionable insights using Gemini 3.")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        if run_analysis:
            with st.spinner("Analyzing data with Gemini..."):
                prompt = build_prompt(df)
                insights = analyze_with_gemini(prompt)

                st.subheader("📊 AI-Generated Insights")
                st.markdown(insights)

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
