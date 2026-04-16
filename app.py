import streamlit as st
import torch
import time
import pandas as pd
import PyPDF2
import docx
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartTokenizer, BartForConditionalGeneration
from textrank_summarizer import TextRankSummarizer

# ==========================================
# Page Config (Must be first Streamlit command)
# ==========================================
st.set_page_config(page_title="Wohoo Text Summarizer", page_icon="📝", layout="wide")

# ==========================================
# Model Loading
# ==========================================
@st.cache_resource
def load_t5():
    tokenizer = AutoTokenizer.from_pretrained("RAINN4439/my_custom_t5")
    model = AutoModelForSeq2SeqLM.from_pretrained("RAINN4439/my_custom_t5")
    return tokenizer, model

@st.cache_resource
def load_bart():
    tokenizer = BartTokenizer.from_pretrained("RAINN4439/my_custom_bart")
    model = BartForConditionalGeneration.from_pretrained("RAINN4439/my_custom_bart")
    return tokenizer, model

@st.cache_resource
def load_textrank():
    # Extracts the top 3 sentences
    return TextRankSummarizer(top_n=3, lemmatize=True, min_sentence_len=5)

# Try loading models to prevent app crash if folders are missing
try:
    t5_tokenizer, t5_model = load_t5()
    t5_ready = True
except Exception:
    t5_ready = False

try:
    bart_tokenizer, bart_model = load_bart()
    bart_ready = True
except Exception:
    bart_ready = False

textrank_summarizer = load_textrank()

# ==========================================
# File Extraction & TEXT CLEANING Helper
# ==========================================
def extract_text_from_file(uploaded_file):
    extracted_text = ""
    try:
        if uploaded_file.name.endswith('.txt'):
            extracted_text = uploaded_file.read().decode('utf-8')
        elif uploaded_file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    extracted_text += text + " "
        elif uploaded_file.name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                if para.text.strip():
                    extracted_text += para.text + " "
        
        clean_text = extracted_text.replace('\n', ' ').replace('\r', ' ')
        # 2. 把多个连续的空格合并成一个正常的空格
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text.strip()
        
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""

# ==========================================
# Sidebar Navigation
# ==========================================
st.sidebar.title("🧭 Wohoo Text Summarizer")
st.sidebar.markdown("---")
page = st.sidebar.radio("Select Module:", ["📝 Smart Summarizer", "📊 Model Metrics Dashboard"])
st.sidebar.markdown("---")

# ==========================================
# Page 1: Smart Summarizer
# ==========================================
if page == "📝 Smart Summarizer":
    st.title("📝 Smart NLP Text Summarizer")
    st.markdown("Upload a document OR paste your text below, then select an AI engine to generate a summary.")

    # UI Layout: 2 Columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Input Selection")
        model_choice = st.selectbox(
            "⚙️ Choose NLP Engine: ",
            ("TextRank", "T5-Small", "BART-Large")
        )
        
        # Hybrid Input Logic
        uploaded_file = st.file_uploader("📂 Upload a document (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"])
        
        default_text = ""
        if uploaded_file is not None:
            default_text = extract_text_from_file(uploaded_file)
            st.success("✅ File loaded and cleaned successfully! You can edit the text below.")
            
        article_text = st.text_area("📄 Paste or edit your article text here: ", value=default_text, height=300)
        generate_btn = st.button("🚀 Execute Inference Pipeline", use_container_width=True)

    with col2:
        st.subheader("Inference Result")
        
        if generate_btn:
            if article_text.strip() == "":
                st.warning("⚠️ Please input text or upload a valid file first!")
            else:
                # Extra safety cleanup for pasted text
                clean_article_text = re.sub(r'\s+', ' ', article_text.replace('\n', ' ')).strip()
                
                with st.spinner(f"Processing via {model_choice}..."):
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    final_summary = ""
                    
                    start_time = time.time()

                    if "T5" in model_choice:
                        if t5_ready:
                            t5_model.to(device)
                            t5_input = "summarize: " + clean_article_text
                            inputs = t5_tokenizer(t5_input, return_tensors="pt", max_length=512, truncation=True).to(device)
                            summary_ids = t5_model.generate(
                                inputs["input_ids"], max_length=120, min_length=50, num_beams=4, early_stopping=True
                            )
                            final_summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        else:
                            st.error("T5 Model not found. Did you finish training?")

                    elif "BART" in model_choice:
                        if bart_ready:
                            bart_model.to(device)
                            inputs = bart_tokenizer(clean_article_text, return_tensors="pt", max_length=512, truncation=True).to(device)
                            summary_ids = bart_model.generate(
                                inputs["input_ids"], max_length=120, min_length=50, num_beams=4, early_stopping=True
                            )
                            final_summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                        else:
                            st.error("BART Model not found. Did you finish training?")

                    elif "TextRank" in model_choice:
                        final_summary = textrank_summarizer.summarize(clean_article_text)
                    
                    end_time = time.time()
                    inference_time = end_time - start_time
                
                # Display Result
                if final_summary:
                    st.success("✅ Summary Generated Successfully!")
                    st.info(final_summary)
                    
                    st.markdown("### ⏱️ Telemetry & Logs")
                    met_col1, met_col2 = st.columns(2)
                    met_col1.metric(label="Inference Speed", value=f"{inference_time:.3f} s")
                    met_col2.metric(label="Cleaned Word Count", value=str(len(clean_article_text.split())))

# ==========================================
# Page 2: Model Metrics Dashboard
# ==========================================
elif page == "📊 Model Metrics Dashboard":
    st.title("📊 Model Metrics & Speed Comparison")
    st.markdown("Performance analysis across **5,000 unseen test articles** based on the final evaluation report.")
    
    st.markdown("### 🏆 High-Level Accuracy (ROUGE-1)")
    m1, m2, m3 = st.columns(3)
    m1.metric("BART-Large", "42.45%", "+2.84% vs T5")
    m2.metric("T5-Small", "39.61%", "+4.14% vs TextRank")
    m3.metric("TextRank", "35.47%", "Baseline")

    st.markdown("---")
    
    st.markdown("### 📋 Comparison Table")
    data = {
        "Model Architecture": ["BART-Large (Heavyweight)", "T5-Small (Lightweight)", "TextRank (Extractive)"],
        "ROUGE-1 (%)": [42.45, 39.61, 35.47],
        "ROUGE-2 (%)": [19.93, 18.14, 14.04],
        "ROUGE-L (%)": [29.13, 28.03, 23.82],
        "Avg Time per Article (s)": [4.23, 1.39, 0.008],
        "Total Test Time (5k articles)": ["~6 Hours", "~1.95 Hours", "43 Seconds"]
    }
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.markdown("### 📈 Visual Comparison")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("**ROUGE Scores Distribution**")
        chart_data = pd.DataFrame(
            {
                "ROUGE-1": [42.45, 39.61, 35.47],
                "ROUGE-2": [19.93, 18.14, 14.04],
                "ROUGE-L": [29.13, 28.03, 23.82]
            },
            index=["BART", "T5", "TextRank"]
        )
        st.bar_chart(chart_data)
        
    with chart_col2:
        st.markdown("**Inference Speed (Seconds / Article) - Lower is Better**")
        speed_data = pd.DataFrame(
            {
                "Seconds": [4.23, 1.39, 0.008]
            },
            index=["BART", "T5", "TextRank"]
        )
        st.bar_chart(speed_data, color="#ff4b4b")
