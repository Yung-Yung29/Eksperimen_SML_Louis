
import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


def load_dataset(path):
    """
    Load raw dataset
    """

    df = pd.read_csv(path)

    print("Shape :", df.shape)

    return df


def preprocessing(df):

    print("\n===== PREPROCESSING =====")

    # Missing Value
    print("Missing Value")
    print(df.isnull().sum())

    # Duplicate
    duplicate = df.duplicated().sum()
    print("\nDuplicate :", duplicate)

    if duplicate > 0:
        df = df.drop_duplicates()

    # Split
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    df_processed = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    df_processed["target"] = y.values

    return df_processed, scaler


def save_dataset(df, output_path):

    df.to_csv(
        output_path,
        index=False
    )

def save_scaler(scaler):

    os.makedirs("artifacts", exist_ok=True)

    joblib.dump(
        scaler,
        "artifacts/scaler.pkl"
    )

def main():

    raw_file = "breast_cancer_raw.csv"

    output_file = "breast_cancer_preprocessing.csv"

    df = load_dataset(raw_file)

    df_processed, scaler = preprocessing(df)

    save_dataset(
        df_processed,
        output_file
    )

    save_scaler(scaler)

    print("\n===== DONE =====")


if __name__ == "__main__":

    main()
