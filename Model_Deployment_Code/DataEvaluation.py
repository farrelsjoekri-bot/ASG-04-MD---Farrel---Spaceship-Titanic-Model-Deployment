from sklearn.metrics import accuracy_score, classification_report

from DataTraining import train_model


def eval_model():
    model, X_test_processed, y_test = train_model()

    y_pred = model.predict(X_test_processed)

    accuracy = accuracy_score(y_test, y_pred)

    print("Model evaluation completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return accuracy


if __name__ == "__main__":
    eval_model()