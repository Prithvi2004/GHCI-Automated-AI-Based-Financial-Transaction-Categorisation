# src/evaluate.py
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import numpy as np

def generate_evaluation_report(y_true, y_pred, categories):
    report = classification_report(y_true, y_pred, target_names=categories, output_dict=True)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred, labels=categories)
    return {
        "macro_f1": macro_f1,
        "per_class_f1": {cat: report[cat]['f1-score'] for cat in categories if cat in report},
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }