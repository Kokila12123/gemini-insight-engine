import streamlit as st
import pandas as pd
import google.generativeai as genai

# ---------------- CONFIG ----------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

st.set_page_config(
    page_title="Gemini Insight Engine",
    layout="wide"
)

# ---------------- UI ----------------
st.title("🔍 Gemini Insight Engine")
st.caption("Upload data. Ask questions. Get decisions.")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

# ---------------- LOGIC ----------------
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("📄 Data Preview")
        st.dataframe(df.head())

        if st.button("Analyze with Gemini"):
            with st.spinner("Gemini is analyzing your data..."):
                model = genai.GenerativeModel("gemini-1.5-pro")

                prompt = f"""
You are a senior data analyst.

Analyze the following dataset:

{df.head(50).to_csv(index=False)}

Provide:
1. Key trends
2. Anomalies or unusual behavior
3. Risks or concerns
4. Clear, actionable recommendations

Explain in simple language.
"""

                response = model.generate_content(prompt)

                st.subheader("📊 Gemini Insights")
                st.write(response.text)

    except Exception as e:
        st.error(f"Error reading file: {e}")
