import streamlit as st
from transformers import pipeline, AutoTokenizer
from typing import Optional

st.set_page_config(page_title="Multilingual Comment Analyzer", layout="wide")

# Cached pipeline dictionaries
_SENTIMENT_PIPELINES: dict = {}
_TRANSLATE_PIPELINES: dict = {}
_TRANSLATE_TOKENIZERS: dict = {}

@st.cache_resource
def get_sentiment_pipeline_cached(model_name: str):
    if model_name not in _SENTIMENT_PIPELINES:
        _SENTIMENT_PIPELINES[model_name] = pipeline(model=model_name)
    return _SENTIMENT_PIPELINES[model_name]

@st.cache_resource
def get_translate_pipeline_and_tokenizer_cached(model_name: str = "Qwen/Qwen3-0.6B"):
    if model_name not in _TRANSLATE_PIPELINES:
        _TRANSLATE_PIPELINES[model_name] = pipeline("text-generation", model=model_name)
    if model_name not in _TRANSLATE_TOKENIZERS:
        _TRANSLATE_TOKENIZERS[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TRANSLATE_PIPELINES[model_name], _TRANSLATE_TOKENIZERS[model_name]

def sentiment(user_input: str, placeholder, sentiment_model_name: str):
    pipeline_obj = get_sentiment_pipeline_cached(sentiment_model_name)
    sentiment_result = pipeline_obj(user_input)
    sentiment_label = sentiment_result[0]["label"]
    confidence = sentiment_result[0]["score"]
    placeholder.markdown(f"**Sentiment ({sentiment_model_name}):** {sentiment_label} ({confidence:.2%} confidence)")

def translate(user_input: str, placeholder, translate_model_name: str = "Qwen/Qwen3-0.6B"):
    translate_pipeline, tokenizer = get_translate_pipeline_and_tokenizer_cached(model_name=translate_model_name)

    content_to_translate = "Translate the following into English: '" + user_input + "'"

    messages = [
        {"role": "user", "content": content_to_translate},
    ]

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
    placeholder.markdown(f"**Meaning in English ({translate_model_name}):** {extracted_response}")

def main():
    st.markdown("## Multilingual Social Media Product Comment Analyzer\n")

    st.markdown(
        """
        <style>
        /* Full viewport height layout with a visible vertical divider */
        .app-split {
            display: flex;
            min-height: 100vh; /* full viewport height */
            gap: 16px;
            align-items: stretch;
        }
        .left-pane {
            flex: 0 0 360px;
            max-width: 420px;
            padding-right: 8px;
            box-sizing: border-box;
        }
        .vertical-divider {
            width: 1px;
            background-color: rgba(0,0,0,0.12);
        }
        .right-pane {
            flex: 1 1 auto;
            padding-left: 16px;
            box-sizing: border-box;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Split layout using Streamlit columns (valid specs only)
    left_col, divider_col, right_col = st.columns([1, 0.02, 3])

    # Left pane content (controls)
    with left_col:
        st.markdown("### Controls")
        st.caption("Change settings here, then click Apply. Use Analyze to run with the active configuration.")

        model_choice = st.selectbox(
            "Translation model",
            options=[
                "Qwen/Qwen3-0.6B",
                # add other translation models here
            ],
            index=0,
            help="Select the model used for translation/generation. Changes take effect after clicking Apply."
        )

        sentiment_choice = st.selectbox(
            "Sentiment model",
            options=[
                "ivanwonghs/multilingual_comment_sentiment_finetuned_on_amazon_reviews_final",
                # add other sentiment models here
            ],
            index=0,
            help="Select the model used for sentiment classification. Changes take effect after clicking Apply."
        )

        function_choice = st.selectbox(
            "Function",
            options=["Both", "Sentiment", "Translate"],
            index=0,
            help="Select which operation(s) to run on the comment. Changes take effect after clicking Apply."
        )

        st.markdown("#### Supported languages")
        st.markdown(
            """
            - English
            - Arabic
            - German
            - Spanish
            - French
            - Japanese
            - Chinese
            - Indonesian
            - Hindi
            - Italian
            - Malay
            - Portuguese
            """
        )

        # Apply button sets the chosen configuration in session state
        if st.button("Apply"):
            st.session_state["applied_translate_model"] = model_choice
            st.session_state["applied_sentiment_model"] = sentiment_choice
            st.session_state["applied_function"] = function_choice
            st.success("Configuration applied. Use Analyze to run with this configuration.")

    # Divider column: draw a full-height divider using HTML/CSS hack
    with divider_col:
        st.markdown(
            """
            <div style="height:100vh; width:1px; background-color:rgba(0,0,0,0.12); margin:0 8px;"></div>
            """,
            unsafe_allow_html=True,
        )

    # Right pane content (input and results)
    with right_col:
        translate_model_applied = st.session_state.get("applied_translate_model", model_choice)
        sentiment_model_applied = st.session_state.get("applied_sentiment_model", sentiment_choice)
        function_applied = st.session_state.get("applied_function", function_choice)

        st.markdown(f"**Active configuration:** Translation model: `{translate_model_applied}` — Sentiment model: `{sentiment_model_applied}` — Function: `{function_applied}`")

        user_input = st.text_input("Please input the comment you want to analyse:")

        # Analyze button runs inference explicitly
        if st.button("Analyze"):
            if not user_input:
                st.warning("Please enter a comment to analyze.")
            else:
                status_placeholder = st.empty()
                sentiment_placeholder = st.empty()
                translate_placeholder = st.empty()

                with st.spinner("Analyzing comment — this may take a while..."):
                    status_placeholder.info("Loading models and running inference. Please wait...")

                    if function_applied in ("Both", "Sentiment"):
                        try:
                            sentiment(user_input, sentiment_placeholder, sentiment_model_name=sentiment_model_applied)
                        except Exception as e:
                            sentiment_placeholder.error(f"Sentiment error: {e}")

                    if function_applied in ("Both", "Translate"):
                        try:
                            translate(user_input, translate_placeholder, translate_model_name=translate_model_applied)
                        except Exception as e:
                            translate_placeholder.error(f"Translate error: {e}")

                    status_placeholder.success("Analysis complete.")

if __name__ == "__main__":
    main()
