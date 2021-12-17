import numpy as np
from joblib import load

class StandardScaler:

    def load(self, scaler_path: str):
        self.scaler = load(scaler_path)

    def transform(self,X_test: np.ndarray)->np.ndarray:
        return self.scaler.transform(X_test)
