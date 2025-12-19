# ============================================================
# Streamlit App: AI-Powered Data Leak Detection System (Working Version)
# Features: Automatic PII detection, Synthetic leaks, IsolationForest ML
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import IsolationForest

sns.set(style="whitegrid")

# ============================================================
# 1. App Title
# ============================================================
st.set_page_config(page_title="ML Data Leak Detector", layout="wide")
st.title("ML-Based Data Leak Detection System")

# ============================================================
# 2. Upload Datasets
# ============================================================
st.sidebar.header("Upload Databases")
master_file = st.sidebar.file_uploader("MASTER Database CSV", type=["csv"])
current_file = st.sidebar.file_uploader("CURRENT Database CSV", type=["csv"])

if master_file is not None and current_file is not None:
    try:
        master_df = pd.read_csv(master_file, dtype=str).fillna("")
        current_df = pd.read_csv(current_file, dtype=str).fillna("")

        st.write("### MASTER DB Sample")
        st.dataframe(master_df.head())
        st.write("### CURRENT DB Sample")
        st.dataframe(current_df.head())

        if "id" not in master_df.columns or "id" not in current_df.columns:
            st.error("Error: 'id' column is required in both CSVs!")
        else:

            # ============================================================
            # # 3. Automatic PII Detection
            # # ============================================================
            # st.subheader("Detecting PII Columns Automatically")
            st.subheader("Detecting PII Columns Automatically")
            SENSITIVE_COLS = []

            sample_rows = current_df.head(100)
            for col in current_df.columns:
                if current_df[col].dtype == object:
                    for val in sample_rows[col]:
                        if isinstance(val, str):
                            if re.match(r".+@.+\..+", val):
                                SENSITIVE_COLS.append(col)
                                break
                            if re.match(r"\+?\d{10,15}", val):
                                SENSITIVE_COLS.append(col)
                                break
                            if re.match(r"\d{3}-\d{2}-\d{4}", val):
                                SENSITIVE_COLS.append(col)
                                break
            SENSITIVE_COLS = list(set(SENSITIVE_COLS))
            st.write("Detected PII Columns:", SENSITIVE_COLS)

            # ============================================================
            # 4. Simulate PII Leaks (optional)
            # ============================================================
            leak_ratio = st.sidebar.slider("Simulate PII Leak (%)", 0, 10, 2)
            def simulate_pii_leaks(df, leak_ratio=0.02):
                df_leak = df.copy()
                n = int(len(df_leak) * leak_ratio / 100)
                if n == 0: return df_leak
                leak_rows = np.random.choice(df_leak.index, n, replace=False)
                for col in SENSITIVE_COLS:
                    if col in df_leak.columns:
                        df_leak.loc[leak_rows, col] = df_leak.loc[leak_rows, col].apply(lambda x: str(x) + "_LEAK")
                return df_leak

            current_df = simulate_pii_leaks(current_df, leak_ratio)
            st.write(f"Simulated {leak_ratio}% PII leaks in CURRENT DB.")

            # ============================================================
            # 5. Feature Engineering
            # ============================================================
            UNIQUE_KEY = "id"
            compare_cols = [c for c in master_df.columns if c != UNIQUE_KEY]

            master_idx = master_df.set_index(UNIQUE_KEY)
            current_idx = current_df.set_index(UNIQUE_KEY)
            all_ids = sorted(list(set(master_idx.index) | set(current_idx.index)))

            records = []
            for rid in all_ids:
                in_master = int(rid in master_idx.index)
                in_current = int(rid in current_idx.index)

                master_row = master_idx.loc[rid].to_dict() if in_master else {}
                curr_row = current_idx.loc[rid].to_dict() if in_current else {}

                diff_count = 0
                pii_diff = 0
                for c in compare_cols:
                    if str(master_row.get(c, "")) != str(curr_row.get(c, "")):
                        diff_count += 1
                        if c in SENSITIVE_COLS:
                            pii_diff += 1

                records.append({
                    UNIQUE_KEY: rid,
                    "missing_flag": 1 if (in_master and not in_current) else 0,
                    "new_flag": 1 if (in_current and not in_master) else 0,
                    "changed_cols": diff_count,
                    "pii_flag": 1 if pii_diff > 0 else 0
                })

            features_df = pd.DataFrame(records).set_index(UNIQUE_KEY)


            # ============================================================
            # 7. IsolationForest ML
            # ============================================================
            iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42)
            X = features_df[["missing_flag","new_flag","changed_cols","pii_flag"]].astype(float)
            iso.fit(X)
            features_df["anomaly_score"] = iso.decision_function(X)
            features_df["anomaly"] = pd.Series(iso.predict(X), index=features_df.index).replace({-1:1,1:0})

            # ============================================================
            # 8. Risk Mapping
            # ============================================================
            def risk_level(row):
                if row["missing_flag"] == 1:
                    return "HIGH – Record Deleted"
                if row["pii_flag"] == 1:
                    return "HIGH – Sensitive PII Modified"
                if row["new_flag"] == 1:
                    return "MEDIUM – New Record Added"
                if row["changed_cols"] > 0:
                    return "MEDIUM – Modified"
                return "LOW"

            features_df["risk"] = features_df.apply(risk_level, axis=1)
            st.subheader("Risk Distribution")
            st.bar_chart(features_df["risk"].value_counts())

            st.subheader("High-Risk Anomalies")
            st.dataframe(features_df[features_df["anomaly"] == 1])

            # ============================================================
            # 9. Download Report
            # ============================================================
            csv = features_df.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Download Anomaly Report CSV",
                data=csv,
                file_name='data_leak_report_pii.csv',
                mime='text/csv',
            )

            st.success("Scan Completed with PII Security Enhancements!")

    except Exception as e:
        st.error(f"Error: {e}")
