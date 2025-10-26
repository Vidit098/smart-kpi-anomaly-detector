import pandas as pd

def prepare_ts(df, date_col="date", kpi_col="revenue", freq="D"):
    # Select required columns
    if date_col not in df.columns or kpi_col not in df.columns:
        raise ValueError(f"Columns not found: {date_col} or {kpi_col}")
    ts = df[[date_col, kpi_col]].copy()
    ts = ts.rename(columns={date_col: "date", kpi_col: "value"})
    ts = ts.set_index("date").sort_index()
    # resample and fill missing periods
    ts = ts.resample(freq).sum()
    ts["value"] = ts["value"].fillna(0)
    # rolling features
    ts["rolling_mean_7"] = ts["value"].rolling(window=7, min_periods=1).mean()
    ts["rolling_std_7"] = ts["value"].rolling(window=7, min_periods=1).std().fillna(0.0)
    ts["z_score"] = (ts["value"] - ts["rolling_mean_7"]) / (ts["rolling_std_7"].replace(0, 1))
    ts = ts.fillna(0)
    return ts