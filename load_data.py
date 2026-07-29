import os
import pandas as pd


DATA_PATH = r"C:\Users\veerendra\PycharmProjects\PythonProject1\placement_predict_50k Dataset (3) 1.csv"


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the CSV dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    return df


def get_data_summary(path: str = DATA_PATH) -> dict:
    """Return summary information about the dataset."""
    df = load_data(path)

    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_counts": {col: int(df[col].isna().sum()) for col in df.columns},
        "preview": df.head(10).to_dict(orient="records"),
    }

    return summary


if __name__ == "__main__":
   
    summary = get_data_summary()
    print(summary)