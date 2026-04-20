CLASS_LABELS = [
    "legitimate",
    "phishing",
    "banking",
    "investment",
    "kidnapping",
    "lottery",
    "customer_service",
    "identity_theft",
]

N_CLASSES = len(CLASS_LABELS)
LABEL_TO_IDX = {label: i for i, label in enumerate(CLASS_LABELS)}
IDX_TO_LABEL = {i: label for i, label in enumerate(CLASS_LABELS)}
