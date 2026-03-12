import pickle
from pathlib import Path

from sklearn.linear_model import LogisticRegression

from DataPreprocessor import building_preprocessor

BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_FILE = ARTIFACTS_DIR / "model.pkl"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.pkl"


def train_model():
    X_train, X_test, y_train, y_test, preprocessor = building_preprocessor()

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_processed, y_train)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("Model trained successfully.")
    print(f"Fitted preprocessor saved to: {PREPROCESSOR_FILE}")
    print(f"Model saved to: {MODEL_FILE}")

    return model, X_test_processed, y_test

if __name__ == "__main__":
    train_model()