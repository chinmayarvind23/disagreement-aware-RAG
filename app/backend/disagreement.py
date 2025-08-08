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

import torch, torch.nn as nn, torch.optim as optim
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

class DisagreeHead:
    def __init__(self, hidden=16):
        self.threshold = 0.3
        self.scaler = StandardScaler()
        self.model = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        ).to("cuda")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, lr=1e-2):
        Xs = self.scaler.fit_transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device="cuda")
        yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device="cuda")
        opt = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            opt.zero_grad()
            pred = self.model(Xt)
            loss_fn(pred, yt).backward()
            opt.step()

    # Predict the probability of disagreement given features
    def predict_proba(self, feats: dict) -> float:
        x = self.scaler.transform([[feats["sc_var"], feats["overlap"], feats["entropy_proxy"]]])
        xt = torch.tensor(x, dtype=torch.float32, device="cuda")
        return float(self.model(xt).cpu().item())

    def save(self, path:str):
        torch.save({
            "model": self.model.state_dict(),
            "scaler": self.scaler
        }, path)

    @classmethod
    def load(cls, path:str):
        ckpt = torch.load(path, map_location="cuda")
        h = cls()
        h.scaler = ckpt["scaler"]
        h.model.load_state_dict(ckpt["model"])
        return h

# Answer ONLY if predicted disagreement is low, overlap between answer and evidence is good, and self-consistency is high
def decision_from_feats(p_disagree: float, feats: dict,
                        tau=0.3, min_overlap=0.45, max_sc=0.2):
    if p_disagree < tau and feats["overlap"] >= min_overlap and feats["sc_var"] <= max_sc:
        return "answer"
    return "abstain"
