import streamlit as st
import pandas as pd
import google.generativeai as genai

# ---------------- CONFIG ----------------
DEMO_MODE = True  # Set False when quota is available

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
            with st.spinner("Analyzing data..."):
                st.subheader("📊 Gemini Insights")

                if DEMO_MODE:
                    st.markdown("""
### 🔍 Key Trends
- Sales increase steadily except for a sharp dip in March.
- Return rates spike significantly in March.

### ⚠️ Anomalies
- March shows unusually high returns despite lower sales.
- This deviates from the overall trend.

### 🚨 Risks
- Possible quality or logistics issues.
- Risk of customer dissatisfaction.

### ✅ Actionable Recommendations
- Investigate March supply chain issues.
- Add stricter quality checks.
- Monitor returns weekly.
""")
                else:
                    model = genai.GenerativeModel("models/gemini-3-flash-preview")

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
                    st.write(response.text)

    except Exception as e:
        st.error(f"Error reading file: {e}")
