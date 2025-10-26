import pandas as pd

def load_csv_from_file(file_like):
    df = pd.read_csv(file_like)
    # try common date column variations
    date_cols = [c for c in df.columns if "date" in c.lower()]
    if len(date_cols) == 0:
        raise ValueError("No date column found. Make sure your CSV has a 'date' column.")
    df = df.rename(columns={date_cols[0]: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        df = df.dropna(subset=["date"])
    return df.sort_values("date")