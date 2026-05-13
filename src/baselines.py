import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM


# =========================================================
# 1. Event 3-Sigma Baseline
# =========================================================
class ThreeSigmaBaseline:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Assume anomaly signal is first column or 1D input
        x = X[:, 0] if X.ndim > 1 else X
        self.mean_ = np.mean(x)
        self.std_ = np.std(x)

    def score(self, X):
        x = X[:, 0] if X.ndim > 1 else X
        z = np.abs((x - self.mean_) / (self.std_ + 1e-8))
        return z  # higher = more anomalous


# =========================================================
# 2. Isolation Forest
# =========================================================
class IsolationForestBaseline:
    def __init__(self, random_state=42):
        self.model = IsolationForest(
            n_estimators=200,
            contamination='auto',
            max_samples='auto',
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X):
        # keep this for compatibility (not for anomaly decision)
        return -self.model.decision_function(X)


# =========================================================
# 3. One-Class SVM
# =========================================================
class OneClassSVMBaseline:
    def __init__(self, kernel='rbf', nu=0.01, gamma='scale'):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X):
        self.model.fit(X)

    def score(self, X):
        return -self.model.decision_function(X)
    
    def predict(self, X):
        return self.model.predict(X)


# =========================================================
# 4. Layered 3-Sigma Baseline
# =========================================================
class LayeredSigmaBaseline:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        # Use all features (layered behavior)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)

    def score(self, X):
        z = np.abs((X - self.mean_) / (self.std_ + 1e-8))
        # aggregate across features
        return np.mean(z, axis=1)


# =========================================================
# 5. Dispersion MAD-Z Baseline
# =========================================================
class MADZBaseline:
    def __init__(self):
        self.median_ = None
        self.mad_ = None

    def fit(self, X):
        # robust center + dispersion
        self.median_ = np.median(X, axis=0)
        self.mad_ = np.median(np.abs(X - self.median_), axis=0)

    def score(self, X):
        z = np.abs((X - self.median_) / (self.mad_ + 1e-8))
        return np.mean(z, axis=1)


# =========================================================
# 6. LSTM Autoencoder Baseline (Wrapper)
# =========================================================
class LSTMAutoencoderBaseline:
    def __init__(self, model):
        """
        model: pre-trained LSTM autoencoder with .predict()
        """
        self.model = model

    def fit(self, X):
        # assume pre-trained OR trained externally
        pass

    def score(self, X):
        preds = self.model.predict(X)

        # reconstruction error
        if preds.shape == X.shape:
            error = np.mean((X - preds) ** 2, axis=1)
        else:
            error = np.ravel(preds)

        return error


# =========================================================
# 7. Transformer Autoencoder Baseline (Wrapper)
# =========================================================
class TransformerAutoencoderBaseline:
    def __init__(self, model):
        """
        model: pre-trained transformer autoencoder
        """
        self.model = model

    def fit(self, X):
        pass

    def score(self, X):
        preds = self.model.predict(X)

        if preds.shape == X.shape:
            error = np.mean((X - preds) ** 2, axis=1)
        else:
            error = np.ravel(preds)

        return error