import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies_iso(ts_df, contamination=0.02):
    X = ts_df[["value","z_score"]].fillna(0).values
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    scores = iso.decision_function(X)  # higher = normal, lower = anomalous
    preds = iso.predict(X)             # -1 anomaly, 1 normal
    ts_df["iso_score"] = -scores       # invert so higher => more anomalous
    ts_df["anomaly_iso"] = (preds == -1).astype(int)
    return ts_df

def detect_anomalies_zscore(ts_df, threshold=3.0):
    ts_df["anomaly_z"] = (ts_df["z_score"].abs() >= threshold).astype(int)
    return ts_df

def detect_anomalies_ensemble(ts_df):
    ts_df = detect_anomalies_zscore(ts_df, threshold=3.0)
    ts_df = detect_anomalies_iso(ts_df, contamination=0.02)
    # combine: mark final anomaly if either method flags it
    ts_df["anomaly_final"] = ((ts_df["anomaly_z"] == 1) | (ts_df["anomaly_iso"] == 1)).astype(int)
    return ts_df
