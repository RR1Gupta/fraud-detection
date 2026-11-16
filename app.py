"""Streamlit web app for the Credit Card Fraud Detection model.  Run:  streamlit run app.py"""
import json, os
import joblib
import pandas as pd
import streamlit as st
from predict_utils import predict_fraud

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load("model/fraud_model.joblib")


@st.cache_data
def load_samples():
    return pd.read_csv("sample_transactions.csv")


def load_results():
    path = "outputs/results.json"
    return json.load(open(path)) if os.path.exists(path) else None


st.title("💳 Credit Card Fraud Detection")
st.caption("Computational Intelligence project - machine learning model deployed as a web app")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model file could not be loaded: {e}. Run `python train.py` first.")
    st.stop()

threshold = st.sidebar.slider("Fraud decision threshold", 0.05, 0.95, 0.5, 0.05,
                              help="Flag a transaction as fraud if its probability is above this value.")
st.sidebar.info("Lower threshold = catches more fraud but raises more false alarms.")

tab1, tab2, tab3 = st.tabs(["Try a sample transaction", "Upload a CSV file", "About the model"])

with tab1:
    try:
        samples = load_samples()
        idx = st.selectbox("Pick a sample transaction from the test set", samples.index,
                           format_func=lambda i: f"Transaction #{i}  (amount = {samples.loc[i, 'Amount']:.2f})")
        row = samples.loc[[idx]]
        res = predict_fraud(row.drop(columns=["Class"]), model, threshold).iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud probability", f"{res['fraud_probability']*100:.1f}%")
        c2.metric("Model decision", res["prediction"])
        c3.metric("Actual label (known)", "FRAUD" if int(row["Class"].iloc[0]) == 1 else "Legitimate")
        if res["prediction"] == "FRAUD":
            st.error("⚠️ This transaction looks fraudulent.")
        else:
            st.success("✅ This transaction looks legitimate.")
        with st.expander("Show the transaction features"):
            st.dataframe(row.drop(columns=["Class"]).T.rename(columns={idx: "value"}))
    except Exception as e:
        st.error(f"Could not score the sample: {e}")

with tab2:
    st.write("Upload a CSV with columns **Time, V1 ... V28, Amount** (like the public dataset). Max 50,000 rows.")
    up = st.file_uploader("CSV file", type=["csv"])
    if up is not None:
        try:
            out = predict_fraud(pd.read_csv(up), model, threshold)
            n_fraud = int((out["prediction"] == "FRAUD").sum())
            a, b, c = st.columns(3)
            a.metric("Transactions checked", len(out)); b.metric("Flagged as fraud", n_fraud)
            c.metric("Fraud rate", f"{100*n_fraud/len(out):.2f}%")
            flagged = out[out["prediction"] == "FRAUD"].sort_values("fraud_probability", ascending=False)
            st.subheader("Flagged transactions")
            st.dataframe(flagged.head(200) if n_fraud else flagged)
            st.download_button("Download all predictions (CSV)", out.to_csv(index=False), "predictions.csv", "text/csv")
        except ValueError as e:
            st.warning(str(e))
        except Exception as e:
            st.error(f"Something went wrong while reading the file: {e}")

with tab3:
    r = load_results()
    if r is None:
        st.info("Run `python train.py` to generate the model report.")
    else:
        st.write(f"**Dataset:** {r['dataset_source']}  |  **Rows:** {r['rows_after_dedup']:,}  |  "
                 f"**Fraud cases:** {r['n_fraud']} ({r['fraud_percent']}%)")
        st.write(f"**Final model:** {r['best_model']}  |  **Best parameters:** {r['best_params']}")
        st.dataframe(pd.DataFrame(r["models"]).set_index("model"))
        for img in ["confusion_matrix.png", "curves.png", "feature_importance.png"]:
            if os.path.exists(f"outputs/{img}"):
                st.image(f"outputs/{img}")
