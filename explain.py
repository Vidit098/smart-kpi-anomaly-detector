def explain_anomaly_row(ts_df, anomaly_date):
    if anomaly_date not in ts_df.index:
        return ["No data for this date."]
    row = ts_df.loc[anomaly_date]
    reasons = []
    # reason 1: z-score large
    if abs(row.get("z_score", 0)) >= 3.0:
        reasons.append(f"Large deviation: z-score = {row['z_score']:.2f} (value {row['value']:.0f} vs 7-period mean {row['rolling_mean_7']:.0f}).")
    # reason 2: IsolationForest score
    if row.get("anomaly_iso", 0) == 1:
        reasons.append(f"Model flag: IsolationForest anomaly (iso_score={row.get('iso_score',0):.4f}).")
    # fallback
    if not reasons:
        reasons.append("Anomaly flagged by ensemble methods; further investigation recommended (check related dimensions).")
    return reasons
