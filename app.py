def format_sentiment(label: str, score: float) -> str:
    """
    label: 'POS' or 'NEG' (case-insensitive)
    score: float between 0 and 1 (model confidence)
    returns string like "Positive (confidence : 24%)"
    """
    label = label.strip().upper()
    if label == "POS" or label == "POSITIVE":
        text = "Positive"
    elif label == "NEG" or label == "NEGATIVE":
        text = "Negative"
    else:
        # fallback to raw label if unknown
        text = label.capitalize()
    percent = round(score * 100)  # integer percent; use int(score*100) if you prefer truncation
    return f"{text} (confidence : {percent}%)"

# usage
print(format_sentiment("POS", 0.235))  # => "Positive (confidence : 24%)"
print(format_sentiment("NEG", 0.141))  # => "Negative (confidence : 14%)"
