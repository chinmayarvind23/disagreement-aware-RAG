import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

# Creating a disagreement head for predicting disagreement risk using logistic regression
class DisagreeHead:
    def __init__(self):
        self.model = None
        self.threshold = 0.3  # abstainance threshold, when p_disagree < threshold, we answer

    # Fit the logistic regression model on features and labels
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(max_iter=200, class_weight="balanced")
        self.model.fit(X, y)

    # Predict the probability of disagreement given features
    def predict_proba(self, feats: dict) -> float:
        x = np.array([[feats["sc_var"], feats["overlap"], feats["entropy_proxy"]]])
        return float(self.model.predict_proba(x)[0, 1])

    # Save the model and threshold to a file
    def save(self, path: str):
        joblib.dump({"model": self.model, "threshold": self.threshold}, path)

    # Load the model and threshold from a file
    @classmethod
    def load(cls, path: str):
        obj = joblib.load(path)
        h = cls()
        h.model = obj["model"]
        h.threshold = obj.get("threshold", 0.3)
        return h

def decision_from_feats(p_disagree: float, feats: dict, tau=0.3, min_overlap=0.4, max_sc=0.5):
    # Answer ONLY if predicted disagreement is low, overlap between answer and evidence is decent, and self-consistency is decent
    if p_disagree < tau and feats["overlap"] >= min_overlap and feats["sc_var"] <= max_sc:
        return "answer"
    return "abstain"