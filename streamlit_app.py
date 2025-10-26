import streamlit as st
import pandas as pd
import plotly.express as px
from src.ingestion import load_csv_from_file
from src.preprocessing import prepare_ts
from src.detection import detect_anomalies_ensemble
from src.explain import explain_anomaly_row

st.set_page_config(page_title="Smart KPI Anomaly Detector", layout="wide")
st.title("Smart KPI Anomaly Detector 🕵️‍♂️📈")

st.markdown(
    """
Upload a CSV with a `date` column and a numeric KPI (e.g. revenue).  
This app runs an ensemble (rolling z-score + IsolationForest) and shows flagged anomalies with simple explanations.
"""
)

uploaded = st.file_uploader("Upload CSV file (date column name: date)", type=["csv"])
st.caption("Sample dataset available in the repo: `data/sample_sales.csv`")

col1, col2 = st.columns([3,1])
with col2:
    freq = st.selectbox("Resample frequency", ["D","W","M"], index=0, help="D=Daily, W=Weekly, M=Monthly")
    kpi_column = st.text_input("KPI column name", value="revenue")
    run_btn = st.button("Run detection")

if uploaded and run_btn:
    try:
        df = load_csv_from_file(uploaded)
        ts = prepare_ts(df, date_col="date", kpi_col=kpi_column, freq=freq)
        ts = detect_anomalies_ensemble(ts)
        st.markdown("### KPI timeseries with anomalies")
        fig = px.line(ts.reset_index(), x="date", y="value", title="KPI over time")
        fig.add_scatter(x=ts[ts["anomaly_final"]==1].index, y=ts[ts["anomaly_final"]==1]["value"],
                        mode="markers", marker=dict(size=10, color="red"), name="Anomaly")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Detected anomalies (most recent first)")
        anomalies = ts[ts["anomaly_final"]==1].sort_index(ascending=False)
        if anomalies.empty:
            st.info("No anomalies detected for the chosen KPI/frequency.")
        else:
            st.dataframe(anomalies[["value","rolling_mean_7","rolling_std_7","z_score","iso_score","anomaly_final"]].reset_index())

            sel = st.selectbox("Select anomaly date to explain", list(anomalies.index.astype(str)))
            if sel:
                sel_idx = pd.to_datetime(sel)
                explanation = explain_anomaly_row(ts, sel_idx)
                st.markdown("**Explanation:**")
                for line in explanation:
                    st.write("- " + line)

        with st.expander("Download results CSV"):
            out_csv = ts.reset_index().to_csv(index=False)
            st.download_button("Download results", data=out_csv, file_name="kpi_anomaly_results.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error running detection: {e}")

if not uploaded:
    st.info("Upload a CSV to start. Example file: data/sample_sales.csv (paste this into the 'data' folder).")