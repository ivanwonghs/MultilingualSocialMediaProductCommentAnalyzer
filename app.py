import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Optional

st.set_page_config(page_title="Multilingual Comment Analyzer", layout="wide")

# Hard-coded model IDs (no user selection, not displayed as internals)
_TRANSLATE_MODEL_ID = "Qwen/Qwen3-0.6B"
_SENTIMENT_MODEL_ID = "ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews"

# Cached pipeline dictionaries (keyed by model id)
_SENTIMENT_PIPELINES: dict = {}
_TRANSLATE_PIPELINES: dict = {}
_TRANSLATE_TOKENIZERS: dict = {}

@st.cache_resource
def get_sentiment_pipeline_cached(model_name: str):
    if model_name not in _SENTIMENT_PIPELINES:
        _SENTIMENT_PIPELINES[model_name] = pipeline(model=model_name)
    return _SENTIMENT_PIPELINES[model_name]

@st.cache_resource
def get_translate_pipeline_and_tokenizer_cached(model_name: str = _TRANSLATE_MODEL_ID):
    if model_name not in _TRANSLATE_PIPELINES:
        _TRANSLATE_PIPELINES[model_name] = pipeline("text-generation", model=model_name)
    if model_name not in _TRANSLATE_TOKENIZERS:
        _TRANSLATE_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINES[model_name], _TRANSLATE_TOKENIZERS[model_name]

def sentiment(user_input: str, placeholder):
    try:
        pipeline_obj = get_sentiment_pipeline_cached(_SENTIMENT_MODEL_ID)
        sentiment_result = pipeline_obj(user_input)
        sentiment_label = sentiment_result[0].get("label", "UNKNOWN")
        confidence = sentiment_result[0].get("score", 0.0)
        placeholder.markdown(f"**Sentiment:** {sentiment_label} ({confidence:.2%} confidence)")
    except Exception:
        placeholder.error("Sentiment analysis failed. Please try again later.")

def translate(user_input: str, placeholder):
    try:
        translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer_cached(_TRANSLATE_MODEL_ID)

        content_to_translate = "Translate the following into English: '" + user_input + "'"

        messages = [
            {"role": "user", "content": content_to_translate},
        ]

        # Use tokenizer's chat template if available; otherwise fall back to raw content
        try:
            text_input = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        except Exception:
            text_input = content_to_translate

        outputs = translate_pipeline(text_input, max_new_tokens=1024)
        generated_text_full = outputs[0].get('generated_text', "")

        marker_end_think = "</think>\n\n"
        start_of_response_idx = generated_text_full.rfind(marker_end_think)
        if start_of_response_idx != -1:
            raw_response = generated_text_full[start_of_response_idx + len(marker_end_think):]
        else:
            raw_response = generated_text_full

        extracted_response = raw_response.strip().strip('"')
        placeholder.markdown(f"**Meaning in English:** {extracted_response}")
    except Exception:
        placeholder.error("Translation failed. Please try again later.")

def main():
    st.markdown("## Multilingual Social Media Product Comment Analyzer\n")

    st.markdown(
        """
        <style>
        /* Full viewport height layout with a centered right pane */
        .app-split {
            display: flex;
            min-height: 100vh;
            gap: 16px;
            align-items: stretch;
        }
        .right-pane {
            flex: 1 1 auto;
            padding: 24px;
            box-sizing: border-box;
            max-width: 1200px;
            margin: 0 auto;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Only the right pane (centered)
    with st.container():
        st.markdown('<div class="app-split">', unsafe_allow_html=True)
        right_col = st.columns([1])[0]

        with right_col:
            # New clear section that tells the user which models/functions are used
            st.markdown("### About this analyzer")
            st.markdown(
                f"- Translation model: `{_TRANSLATE_MODEL_ID}`  \n"
                f"- Sentiment model: `{_SENTIMENT_MODEL_ID}`  \n"
                "- Function: Both (Sentiment + Translate)"
            )
            st.write("")  # spacing

            st.markdown(
                """
                Enter a comment below and click Analyze. The app will return a sentiment label with confidence and an English translation/meaning.
                """
            )

            user_input = st.text_input("Please input the comment you want to analyse:")

            if st.button("Analyze"):
                if not user_input:
                    st.warning("Please enter a comment to analyze.")
                else:
                    status_placeholder = st.empty()
                    sentiment_placeholder = st.empty()
                    translate_placeholder = st.empty()

                    with st.spinner("Analyzing comment — this may take a while..."):
                        status_placeholder.info("Loading models and running inference. Please wait...")

                        # Run sentiment
                        sentiment(user_input, sentiment_placeholder)

                        # Run translation
                        translate(user_input, translate_placeholder)

                        status_placeholder.success("Analysis complete.")

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
