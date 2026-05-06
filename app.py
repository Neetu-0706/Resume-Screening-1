import streamlit as st
import os
import re
import numpy as np
import PyPDF2
import json

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# ---------------- CONFIG ----------------
os.environ["OPENAI_API_KEY"] = ""

st.set_page_config(layout="wide")
st.title("📄 AI Resume Matcher (Dynamic + Explainable)")

# ---------------- PDF READER ----------------
def extract_text_from_pdf(file):
    text = ""
    pdf = PyPDF2.PdfReader(file)
    for page in pdf.pages:
        text += page.extract_text() or ""
    return text.lower()

# ---------------- CLEAN ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip().lower()

# ---------------- COSINE SIM ----------------
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---------------- LLM INIT ----------------
llm = ChatOpenAI(temperature=0)
embeddings = OpenAIEmbeddings()

# ---------------- DYNAMIC SKILL EXTRACTION ----------------
def extract_skills_llm(text, label):
    prompt = f"""
    Extract important skills, tools, technologies from the following {label}.
    Return only JSON list.

    Text:
    {text}
    """

    response = llm.invoke(prompt).content  # ✅ FIXED

    try:
        skills = json.loads(response)
    except:
        skills = []

    return list(set([s.lower() for s in skills]))

# ---------------- SEMANTIC MATCH ----------------
def semantic_match(jd_skills, resume_skills):
    matched = []
    jd_vecs = embeddings.embed_documents(jd_skills)
    res_vecs = embeddings.embed_documents(resume_skills)

    for i, jd_vec in enumerate(jd_vecs):
        best_sim = 0
        best_skill = None

        for j, res_vec in enumerate(res_vecs):
            sim = cosine_similarity(jd_vec, res_vec)
            if sim > best_sim:
                best_sim = sim
                best_skill = resume_skills[j]

        if best_sim > 0.7:
            matched.append((jd_skills[i], best_skill, round(best_sim, 2)))

    return matched

# ---------------- FULL TEXT SIM ----------------
def full_text_similarity(jd, resume):
    vec1 = embeddings.embed_query(jd)
    vec2 = embeddings.embed_query(resume)
    return cosine_similarity(vec1, vec2)

# ---------------- EXPERIENCE MATCH ----------------
def experience_match(jd, resume):
    prompt = f"""
    Compare experience requirement and candidate experience.

    JD: {jd}
    Resume: {resume}

    Return JSON:
    {{
      "score": (0-1),
      "reason": ""
    }}
    """

    response = llm.invoke(prompt).content  # ✅ FIXED

    try:
        data = json.loads(response)
        return data["score"], data["reason"]
    except:
        return 0.5, "Could not determine"

# ---------------- BIAS DETECTION ----------------
def detect_bias_llm(jd):
    prompt = f"""
    Analyze this job description for bias.

    Check:
    - Gender bias
    - Age bias
    - Cultural bias

    Return JSON:
    {{
      "bias_found": true/false,
      "type": [],
      "severity": "low/medium/high",
      "explanation": ""
    }}

    JD:
    {jd}
    """

    response = llm.invoke(prompt).content  # ✅ FIXED

    try:
        return json.loads(response)
    except:
        return {"bias_found": False, "type": [], "severity": "low", "explanation": "N/A"}

# ---------------- UI ----------------
jd_text = st.text_area("📌 Paste Job Description")
resume_file = st.file_uploader("📤 Upload Resume (PDF)", type=["pdf"])

if st.button("🔍 Analyze"):

    if not jd_text or not resume_file:
        st.warning("Please provide both inputs")
        st.stop()

    # Extract
    resume_text = extract_text_from_pdf(resume_file)

    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    st.info("⏳ Extracting skills dynamically...")

    # Dynamic skills
    jd_skills = extract_skills_llm(jd_clean, "job description")
    resume_skills = extract_skills_llm(resume_clean, "resume")

    # Limit size
    jd_skills = jd_skills[:30]
    resume_skills = resume_skills[:50]

    # Semantic match
    matched = semantic_match(jd_skills, resume_skills)

    skill_score = len(matched) / len(jd_skills) if jd_skills else 0

    # Full similarity
    semantic_score = full_text_similarity(jd_clean, resume_clean)

    # Experience
    exp_score, exp_reason = experience_match(jd_clean, resume_clean)

    # Final weighted score
    final_score = (
        (skill_score * 0.5) +
        (semantic_score * 0.3) +
        (exp_score * 0.2)
    ) * 100

    # Bias detection
    bias = detect_bias_llm(jd_clean)

    # Missing skills
    matched_jd = [m[0] for m in matched]
    missing = list(set(jd_skills) - set(matched_jd))

    # Explanation
    explanation_prompt = f"""
    Explain ATS evaluation:

    Score: {final_score}
    Matched Skills: {matched}
    Missing Skills: {missing}
    Experience: {exp_reason}
    Bias: {bias}

    Give clear explanation and improvement suggestions.
    """

    explanation = llm.invoke(explanation_prompt).content  # ✅ FIXED

    # ---------------- OUTPUT ----------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Final Match Score")
        st.metric("Score", f"{round(final_score,2)}%")

        st.subheader("✅ Matched Skills")
        st.write(matched)

    with col2:
        st.subheader("❌ Missing Skills")
        st.write(missing)

        st.subheader("⚠️ Bias Analysis")
        st.write(bias)

    st.subheader("🤖 Explanation")
    st.write(explanation)