import streamlit as st
import pandas as pd
from agents.detector_agent import FraudDetectorAgent
from agents.analyst_agent import AnalystAgent
from agents.alert_agent import AlertAgent
from agents.routing_agent import RoutingAgent
from config.settings import MODEL_NAME

# ------------- PAGE CONFIG ------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# --------- HEADER ---------------------------------------------
st.title("🔍 Agentic Fraud Detection System")
st.caption("Built by Ramya | Project A — Hybrid ML + LLM Pipeline")
st.divider()

# ---------- AGENT INITIALIZATION ------------------------------
@st.cache_resource
def load_agents():
    detector = FraudDetectorAgent(name="DetectorAgent-1")
    analyst  = AnalystAgent(name="AnalystAgent-1")
    alert    = AlertAgent(name="AlertAgent-1")
    router   = RoutingAgent(name="RoutingAgent-1")
    return detector, analyst, alert, router

detector, analyst, alert, router = load_agents()

# --------- SIDE BAR -------------------------------------------
with st.sidebar:
    st.header("System Info")
    st.success(f"Model: {MODEL_NAME}")
    st.info("Hybrid ML + LLM Pipeline Active")

    st.markdown("**Detection Layers:**")
    st.markdown("1. 📋 Rule-based pre-filter")
    st.markdown("2. 🌲 Random Forest scorer (V1–V28)")
    st.markdown("3. 🤖 LLM reasoning (borderline only)")

    st.divider()
    st.markdown("**Agents:**")
    st.markdown("- DetectorAgent — hybrid risk screening")
    st.markdown("- AnalystAgent — investigation")
    st.markdown("- AlertAgent — bank notifications")
    st.markdown("- RoutingAgent — department routing")

    st.divider()
    st.markdown("**Dataset:**")
    st.markdown("- 284,807 transactions")
    st.markdown("- Kaggle Credit Card Fraud")
    st.markdown("- 0.17% fraud rate")

    st.divider()
    st.markdown("**Hybrid Thresholds:**")
    st.markdown("- Score ≥ 0.7 → HIGH (BLOCK)")
    st.markdown("- Score 0.3–0.7 → LLM decides")
    st.markdown("- Score ≤ 0.3 → LOW (APPROVE)")

# ── CONTEXT FOR AGENTS ───────────────────────────────────────
df_full = pd.read_csv("data/transactions_balanced.csv")
avg_amount = df_full["Amount"].mean()
max_amount = df_full["Amount"].max()
context = (f"Dataset average transaction amount: ${avg_amount:.2f}. "
           f"Maximum amount: ${max_amount:.2f}. "
           f"Typical legitimate transactions range from $5 to $500.")

# ── INPUT TABS ────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📂 Upload CSV", "✏️ Manual Entry"])

# ── TAB 1: CSV UPLOAD ─────────────────────────────────────────
with tab1:
    st.subheader("Analyse transactions from CSV file")
    st.caption("Upload a CSV with columns: Amount, hour, Time (V1–V28 optional but recommended)")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(df)} transactions")

        # Show which feature columns are present
        pca_cols = [f"V{i}" for i in range(1, 29)]
        found_pca = [c for c in pca_cols if c in df.columns]
        if found_pca:
            st.info(f"✓ PCA features detected: {len(found_pca)} columns (V1–V28) — ML scorer will use full signal")
        else:
            st.warning("⚠ No V1–V28 columns found — ML scorer will use Amount, Time, hour only")

        st.dataframe(df.head(), use_container_width=True)

        max_txns = min(len(df), 20)
        num_txns = st.slider(
            "Number of transactions to analyse",
            min_value=1,
            max_value=max_txns,
            value=min(5, max_txns)
        )

        if st.button("🔍 Analyse Transactions", type="primary"):
            results = []
            correct = 0

            progress    = st.progress(0)
            status_text = st.empty()

            for i, (_, row) in enumerate(df.head(num_txns).iterrows()):
                transaction = row.to_dict()
                status_text.text(f"Analysing transaction {i+1} of {num_txns}...")
                progress.progress((i + 1) / num_txns)

                # Run detector (rules → ML → LLM)
                response  = detector.analyse(transaction, context)

                # UPDATED THRESHOLD: MEDIUM and HIGH both count as FRAUD
                predicted = "FRAUD" if ("HIGH" in response or "MEDIUM" in response) else "LEGITIMATE"

                # Get actual label if available
                actual = None
                if "is_Fraud" in transaction:
                    actual = "FRAUD" if transaction["is_Fraud"] == 1 else "LEGITIMATE"
                    if predicted == actual:
                        correct += 1

                # Run analyst + router + alert if fraud detected
                investigation = ""
                alert_text    = ""
                routing       = ""
                if predicted == "FRAUD":
                    investigation = analyst.investigate(transaction, response)
                    routing       = router.route(transaction, investigation)
                    if "BLOCK" in investigation:
                        alert_text = alert.generate_alert(
                            transaction, response, investigation
                        )

                results.append({
                    "Amount ($)":  transaction.get("Amount", 0),
                    "Hour":        int(transaction.get("hour", 0)),
                    "Predicted":   predicted,
                    "Actual":      actual if actual else "Unknown",
                    "Risk":        response.split('\n')[0],
                    "Action":      ("BLOCK"    if "BLOCK"    in investigation else
                                   "ESCALATE" if "ESCALATE" in investigation else
                                   "APPROVE"),
                    "Routing":     routing.split('\n')[0] if routing else "—"
                })

            progress.empty()
            status_text.empty()

            # ── Results table ─────────────────────────────────
            st.divider()
            st.subheader("Results")
            results_df = pd.DataFrame(results)

            def highlight_fraud(val):
                if val == "FRAUD":
                    return "background-color: #ff4444; color: white"
                elif val == "LEGITIMATE":
                    return "background-color: #44bb44; color: white"
                return ""

            styled_df = results_df.style.applymap(
                highlight_fraud, subset=["Predicted"]
            )
            st.dataframe(styled_df, use_container_width=True)

            # ── Metrics ───────────────────────────────────────
            if any(r["Actual"] != "Unknown" for r in results):
                st.divider()
                st.subheader("Performance Metrics")

                fraud_results = [r for r in results if r["Actual"] == "FRAUD"]
                legit_results = [r for r in results if r["Actual"] == "LEGITIMATE"]

                tp = sum(1 for r in fraud_results if r["Predicted"] == "FRAUD")
                fn = sum(1 for r in fraud_results if r["Predicted"] == "LEGITIMATE")
                tn = sum(1 for r in legit_results if r["Predicted"] == "LEGITIMATE")
                fp = sum(1 for r in legit_results if r["Predicted"] == "FRAUD")

                precision = round(tp / (tp + fp) * 100, 1) if (tp + fp) > 0 else 0
                recall    = round(tp / (tp + fn) * 100, 1) if (tp + fn) > 0 else 0
                accuracy  = round(correct / num_txns * 100, 1)

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Accuracy",    f"{accuracy}%")
                col2.metric("Precision",   f"{precision}%")
                col3.metric("Recall",      f"{recall}%")
                col4.metric("Fraud Caught", f"{tp}/{tp+fn}")
                col5.metric("False Alarms", f"{fp}")

                # Confusion matrix summary
                with st.expander("Confusion Matrix"):
                    cm_data = {
                        "":                ["Predicted FRAUD", "Predicted LEGIT"],
                        "Actual FRAUD":    [f"✓ {tp} (TP)",   f"✗ {fn} (FN)"],
                        "Actual LEGIT":    [f"✗ {fp} (FP)",   f"✓ {tn} (TN)"]
                    }
                    st.table(pd.DataFrame(cm_data).set_index(""))

            # ── Download ──────────────────────────────────────
            st.divider()
            st.download_button(
                label="⬇ Download Results as CSV",
                data=results_df.to_csv(index=False),
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )

# ── TAB 2: MANUAL ENTRY ───────────────────────────────────────
with tab2:
    st.subheader("Analyse a single transaction manually")
    st.caption("Enter transaction details to get instant fraud assessment through the full 4-agent pipeline")

    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount ($)",
            min_value=0.0,
            max_value=100000.0,
            value=250.0,
            step=0.01
        )
        hour = st.slider(
            "Hour of transaction (0=midnight, 23=11pm)",
            min_value=0,
            max_value=23,
            value=14
        )

    with col2:
        st.info(f"Amount: ${amount:.2f}")
        st.info(f"Hour: {hour}:00 {'AM' if hour < 12 else 'PM'}")

        # Live risk preview
        if amount == 0:
            st.warning("⚠ Zero amount — likely card verification attack")
        elif hour <= 5 and amount > 1000:
            st.warning("⚠ Large overnight transaction — high risk")
        elif hour <= 5:
            st.warning("⚠ Late night transaction — higher risk")
        elif amount > 500:
            st.warning("⚠ High amount — will be scrutinised")
        else:
            st.success("✓ Normal transaction parameters")

    if st.button("🔍 Analyse This Transaction", type="primary"):
        transaction = {
            "Amount": amount,
            "hour":   hour,
            "Time":   50000
        }

        with st.spinner("Running through hybrid detection pipeline..."):

            # Layer 1–3: Detector (rules → ML → LLM)
            response  = detector.analyse(transaction, context)

            # UPDATED THRESHOLD: MEDIUM and HIGH both count as FRAUD
            predicted = "FRAUD" if ("HIGH" in response or "MEDIUM" in response) else "LEGITIMATE"

            st.divider()

            # Result banner
            if predicted == "FRAUD":
                st.error("🚨 FRAUD DETECTED")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")

            # Detector report
            with st.expander("🔍 Detector Agent Report", expanded=True):
                for line in response.strip().split('\n'):
                    if line.strip():
                        st.text(line)

            # Analyst + Router + Alert
            if predicted == "FRAUD":
                investigation = analyst.investigate(transaction, response)

                with st.expander("🔬 Analyst Agent Investigation"):
                    st.text(investigation)

                routing = router.route(transaction, investigation)
                with st.expander("🗺 Routing Decision"):
                    st.text(routing)

                if "BLOCK" in investigation:
                    alert_text = alert.generate_alert(
                        transaction, response, investigation
                    )
                    with st.expander("🚨 Formal Bank Alert", expanded=True):
                        st.code(alert_text)
