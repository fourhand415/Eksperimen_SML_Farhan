import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # Mapping target ke numerik
    df["Heart Disease"] = df["Heart Disease"].map({
        "Absence": 0,
        "Presence": 1
    })

    X = df.drop("Heart Disease", axis=1)
    y = df["Heart Disease"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean["Heart Disease"] = y.values

    return df_clean


def save_data(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]

    raw_path = BASE_DIR / "HeartDisease_Raw.csv"
    output_path = BASE_DIR / "preprocessing" / "heart_disease_preprocessing.csv"

    df_raw = load_data(raw_path)
    df_clean = preprocess_data(df_raw)
    save_data(df_clean, output_path)

    print("✅ Preprocessing selesai")
    print("📁 Output:", output_path)