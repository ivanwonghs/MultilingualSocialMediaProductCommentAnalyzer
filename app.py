# sentiment_example.py
from transformers import pipeline
from typing import Dict, List

def format_sentiment(label: str, score: float) -> str:
    """
    Convert pipeline output (label and score) to the requested string format.
    Examples:
      format_sentiment("POSITIVE", 0.235) -> "Positive (confidence : 24%)"
      format_sentiment("NEGATIVE", 0.141) -> "Negative (confidence : 14%)"
    """
    lab = label.strip().upper()
    if lab.startswith("POS"):
        text = "Positive"
    elif lab.startswith("NEG"):
        text = "Negative"
    else:
        # fallback for unknown labels (capitalize nicely)
        text = label.capitalize()
    percent = round(score * 100)  # round to nearest integer percent
    return f"{text} (confidence : {percent}%)"

def analyze_texts(texts: List[str], model: str = "distilbert-base-uncased-finetuned-sst-2-english") -> List[Dict]:
    """
    Run the Hugging Face sentiment-analysis pipeline on a list of texts.
    Returns list of dicts containing original text and formatted result.
    Default model: a small DistilBERT fine-tuned on SST-2 (binary sentiment).
    """
    nlp = pipeline("sentiment-analysis", model=model, truncation=True)
    raw_results = nlp(texts)
    results = []
    for text, item in zip(texts, raw_results):
        # pipeline returns e.g. {"label": "POSITIVE", "score": 0.9876}
        formatted = format_sentiment(item["label"], float(item["score"]))
        results.append({"text": text, "label": item["label"], "score": float(item["score"]), "display": formatted})
    return results

def main():
    # Example inputs — replace with your actual inputs or read from file / ui
    examples = [
        "I absolutely loved the food and the service was great!",
        "The place was dirty and the staff were rude.",
        "It was okay — not the best, not the worst.",
        "Amazing experience, will definitely come back!",
        "Terrible. I will never return."
    ]

    print("Running sentiment analysis on examples...\n")
    outputs = analyze_texts(examples)

    for out in outputs:
        print("Text :", out["text"])
        print("Result :", out["display"])
        print("-" * 60)

if __name__ == "__main__":
    main()
