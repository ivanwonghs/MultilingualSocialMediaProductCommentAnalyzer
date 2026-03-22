import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer
from typing import Optional
import io

# Lazy global cache for pipelines/tokenizer so we don't reload on every call
_SENTIMENT_PIPELINE: Optional[object] = None
_TRANSLATE_PIPELINE: Optional[object] = None
_TRANSLATE_TOKENIZER: Optional[object] = None

def get_sentiment_pipeline():
    global _SENTIMENT_PIPELINE
    if _SENTIMENT_PIPELINE is None:
        _SENTIMENT_PIPELINE = pipeline(model="ivanwonghs/trial_1")
    return _SENTIMENT_PIPELINE

def get_translate_pipeline_and_tokenizer():
    global _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER
    model_name = "Qwen/Qwen3-0.6B"
    if _TRANSLATE_PIPELINE is None:
        _TRANSLATE_PIPELINE = pipeline("text-generation", model=model_name)
    if _TRANSLATE_TOKENIZER is None:
        _TRANSLATE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINE, _TRANSLATE_TOKENIZER

def analyze_single_comment(user_input: str):
    sentiment_pipeline = get_sentiment_pipeline()
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer()

    # Sentiment
    sent_res = sentiment_pipeline(user_input)
    sentiment_label = sent_res[0]["label"]
    confidence = sent_res[0]["score"]

    # Translation / meaning
    messages = [
        {"role": "user", "content": "Just give me '"+user_input+"' in English purely in string charater"},
    ]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    outputs = translate_pipeline(text_input, max_new_tokens=32768)
    generated_text_full = outputs[0].get('generated_text', "")
    marker_end_think = "</think>\n\n"
    start_of_response_idx = generated_text_full.rfind(marker_end_think)
    if start_of_response_idx != -1:
        raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
    else:
        raw_response = generated_text_full
    extracted_response = raw_response.strip().strip('"')

    return sentiment_label, confidence, extracted_response

def analyze_batch(df: pd.DataFrame, comment_col: str):
    """
    Expects a DataFrame and name of the column that contains comments.
    Returns the DataFrame with added columns: sentiment, confidence, english_meaning
    """
    results = []
    n = len(df)
    sentiment_pipeline = get_sentiment_pipeline()
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer()

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, comment in enumerate(df[comment_col].astype(str).tolist()):
        status_text.text(f"Processing {i+1}/{n}...")
        # Sentiment
        sent_res = sentiment_pipeline(comment)
        sentiment_label = sent_res[0]["label"]
        confidence = sent_res[0]["score"]

        # Translation
        messages = [{"role": "user", "content": "Just give me '"+comment+"' in English purely in string charater"}]
        text_input = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        outputs = translate_pipeline(text_input, max_new_tokens=32768)
        generated_text_full = outputs[0].get('generated_text', "")
        marker_end_think = "</think>\n\n"
        start_of_response_idx = generated_text_full.rfind(marker_end_think)
        if start_of_response_idx != -1:
            raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
        else:
            raw_response = generated_text_full
        extracted_response = raw_response.strip().strip('"')

        results.append({
            "comment": comment,
            "sentiment": sentiment_label,
            "confidence": confidence,
            "english_meaning": extracted_response
        })

        progress_bar.progress((i + 1) / n)

    status_text.success("Batch processing complete.")
    progress_bar.empty()
    return pd.DataFrame(results)

def main():
    st.title("Multilingual Comment Analyzer")
    st.write("AI-powered tool for analyzing multilingual product comments on social media. Detects sentiment and provides English translations/meanings.")

    # Single-comment mode
    st.header("Single Comment")
    user_input = st.text_input("Enter comment here")
    if user_input:
        with st.spinner("Analyzing comment — this may take a while..."):
            sentiment_label, confidence, extracted_response = analyze_single_comment(user_input)
        st.markdown(f"**Sentiment:** {sentiment_label}\n\n**Confidence:** {confidence:.2f}")
        st.markdown(f"**Meaning in English:** {extracted_response}")

    st.markdown("---")

    # Batch mode
    st.header("Batch Mode (CSV or TXT)")
    st.write("Upload a CSV with a column named 'comment' or upload a TXT file with one comment per line.")
    uploaded_file = st.file_uploader("Upload CSV or TXT for batch analysis", type=["csv", "txt"])

    if uploaded_file is None:
        st.info("No batch file uploaded yet. Upload a CSV or TXT to enable Batch Mode.")
    else:
        # Read the uploaded file
        try:
            if uploaded_file.type == "text/csv" or str(uploaded_file.name).lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded CSV:")
                st.dataframe(df.head())
                if "comment" not in df.columns:
                    st.error("CSV must contain a column named 'comment'. Please rename the column containing text to 'comment'.")
                    return
                comment_col = "comment"
            else:
                # treat as text file: each line is a comment
                t = uploaded_file.getvalue().decode("utf-8")
                lines = [line.strip() for line in t.splitlines() if line.strip()]
                df = pd.DataFrame({"comment": lines})
                st.write(f"Loaded {len(lines)} comments from TXT.")
                st.dataframe(df.head())

            # Add an Analyze button to start batch processing
            if st.button("Analyze batch"):
                # Run batch analysis
                with st.spinner("Running batch analysis — this may take a while..."):
                    result_df = analyze_batch(df, comment_col)

                st.success("Batch analysis finished.")
                st.dataframe(result_df)

                # Provide download as CSV
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")
                st.download_button("Download results as CSV", data=csv_bytes, file_name="batch_results.csv", mime="text/csv")
            else:
                st.info("Ready to analyze. Press the 'Analyze batch' button to start processing the uploaded file.")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

if __name__ == "__main__":
    main()
