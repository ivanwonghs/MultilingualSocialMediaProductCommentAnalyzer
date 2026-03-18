from typing import Union, Mapping, Tuple

def format_sentiment(label: Union[str, int], score: float) -> str:
    """
    Convert model output (label and score) to the format:
      "Positive (confidence : 24%)" or "Negative (confidence : 14%)"

    label: str like "POS"/"POSITIVE"/"NEG"/"NEGATIVE" or int like 1/0
    score: float in [0,1] representing confidence for the predicted class

    Returns the formatted string.
    """
    # Normalize label for checking
    if isinstance(label, int):
        lab_norm = label
    else:
        lab_norm = str(label).strip().upper()

    if lab_norm == 1 or (isinstance(lab_norm, str) and lab_norm.startswith("POS")):
        text = "Positive"
    elif lab_norm == 0 or (isinstance(lab_norm, str) and lab_norm.startswith("NEG")):
        text = "Negative"
    else:
        # fallback: try to parse numeric string
        try:
            ln = int(lab_norm)  # may raise
            text = "Positive" if ln == 1 else ("Negative" if ln == 0 else str(label))
        except Exception:
            text = str(label).capitalize()

    percent = round(float(score) * 100)
    return f"{text} (confidence : {percent}%)"


def extract_prediction_from_probs(prob_map: Mapping[str, float], positive_keys: Tuple[str, ...] = ("POS", "POSITIVE", "1"), negative_keys: Tuple[str, ...] = ("NEG", "NEGATIVE", "0")) -> Tuple[Union[str,int], float]:
    """
    When your model returns a probability map (e.g., {"NEG": 0.3, "POS": 0.7} or {"0":0.3, "1":0.7}),
    this helper returns (pred_label, pred_score) where pred_label is either the original key
    (string) or integer 0/1 when possible, and pred_score is the probability of that predicted label.

    Parameters:
    - prob_map: mapping from label/key to probability (values should sum to ~1)
    - positive_keys / negative_keys: keys considered positive/negative (checked case-insensitively)

    Returns:
    - (pred_label, pred_score)
      pred_label: str or int (if key is "0" or "1" or integer)
      pred_score: float in [0,1]
    """
    if not prob_map:
        raise ValueError("prob_map is empty")

    # pick the key with max probability
    best_key = max(prob_map, key=lambda k: float(prob_map[k]))
    best_score = float(prob_map[best_key])

    # try to normalize best_key to int if it's "0" or "1"
    key_norm = best_key
    try:
        if isinstance(best_key, str) and best_key.isdigit():
            key_int = int(best_key)
            if key_int in (0, 1):
                key_norm = key_int
    except Exception:
        key_norm = best_key

    return key_norm, best_score
