from pathlib import Path
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "ingested" / "train.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PREPROCESSOR_FILE = ARTIFACTS_DIR / "preprocessor.pkl"

def spliting_cabin(df):
    cabin_split = df["Cabin"].fillna("Unknown/0/Unknown").str.split("/", expand=True)
    df["CabinDeck"] = cabin_split[0]
    df["CabinNum"] = pd.to_numeric(cabin_split[1], errors="coerce")
    df["CabinSide"] = cabin_split[2]
    return df

def building_preprocessor():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    df = spliting_cabin(df)

    feature_columns = [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinNum",
        "CabinSide",
        "Destination",
        "Age",
        "VIP",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    target_column = "Transported"

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    categorical_features = [
        "HomePlanet",
        "CryoSleep",
        "CabinDeck",
        "CabinSide",
        "Destination",
        "VIP"
    ]

    numerical_features = [
        "CabinNum",
        "Age",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numerical_transformer, numerical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with open(PREPROCESSOR_FILE, "wb") as f:
        pickle.dump(preprocessor, f)

    print("Preprocessing file created successfully.")
    print(f"Preprocessor saved to: {PREPROCESSOR_FILE}")

    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    building_preprocessor()