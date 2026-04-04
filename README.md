# 🔍 Fraud Detection Multi-Agent System
**Project A — Agentic AI Portfolio | Built by Ramya | 2026**

A production-grade fraud detection system combining a hybrid ML + LLM pipeline with a 4-agent architecture. Built as part of the IITM Pravartak Advanced PG Certificate in Agentic AI.

---

## 🏗 Architecture

```
Transaction Input
        │
        ▼
┌─────────────────────┐
│  Rule-based Filter  │  ← Zero-dollar, micro-txn at night, large overnight
└─────────────────────┘
        │ no rule fired
        ▼
┌─────────────────────┐
│  ML Scorer          │  ← Random Forest on V1–V28 + Amount + Time + hour
│  (Random Forest)    │    score ≥ 0.7 → HIGH (BLOCK)
│                     │    score ≤ 0.3 → LOW  (APPROVE)
│                     │    score 0.3–0.7 → BORDERLINE → LLM
└─────────────────────┘
        │ borderline only
        ▼
┌─────────────────────┐
│  LLM DetectorAgent  │  ← GPT reasons on Amount, hour, Time + ML score hint
│  (GPT-4)            │
└─────────────────────┘
        │
        ▼
   FRAUD / LEGIT
        │
   if FRAUD
        ▼
┌─────────────────────┐
│  AnalystAgent       │  ← Deep investigation, recommends BLOCK / ESCALATE
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  RoutingAgent       │  ← Routes to Fraud Dept / IT / General Support
└─────────────────────┘
        │ if BLOCK
        ▼
┌─────────────────────┐
│  AlertAgent         │  ← Generates formal bank alert notification
└─────────────────────┘
```

---

## 📈 Performance — Metrics Progression

| Version | Recall | Precision | Accuracy | Notes |
|---|---|---|---|---|
| LLM only (10 txns) | 60% | 75% | 70% | Baseline |
| LLM only (50 txns) | 32% | 88.9% | 64% | Scaled up |
| + Rules + MEDIUM threshold | 44% | 84.6% | 68% | Improvements 1–3 |
| **Hybrid RF + LLM (50 txns)** | **100%** | **100%** | **100%** | **Final** |

**Key insight:** LLM-only detection on PCA features has an inherent ceiling — the LLM cannot interpret abstract PCA numbers the way a trained classifier can. The hybrid approach uses each tool for what it does best: ML for pattern recognition, LLM for explainability and borderline reasoning.

---

## 🗂 Project Structure

```
fraud-detection/
├── agents/
│   ├── detector_agent.py     # Hybrid: Rules → ML → LLM
│   ├── analyst_agent.py      # Deep investigation
│   ├── alert_agent.py        # Formal bank alert generation
│   └── routing_agent.py      # Department routing
├── tools/
│   ├── ml_scorer.py          # Random Forest scorer (V1–V28)
│   └── image_input.py        # GPT-4V image transaction extraction
├── config/
│   └── settings.py           # API keys, model name, thresholds
├── data/
│   ├── creditcard.csv        # Original Kaggle dataset (284,807 txns)
│   ├── transactions_balanced.csv  # Balanced dataset (984 txns, all features)
│   └── prepare_data.py       # Script to rebuild balanced dataset
├── tests/
│   └── test_agents.py        # pytest test suite (6 tests)
├── app.py                    # Streamlit dashboard
├── main.py                   # CLI pipeline runner
└── requirements.txt
```

---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Ramya192/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Prepare the dataset
Download `creditcard.csv` from [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/`. Then run:
```bash
python data/prepare_data.py
```
This rebuilds `transactions_balanced.csv` with all V1–V28 features preserved.

### 5. Run the CLI pipeline
```bash
python main.py
```

### 6. Launch the Streamlit dashboard
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
langchain
langchain-openai
openai
streamlit
pandas
scikit-learn>=1.6.0
python-dotenv
pytest
```

---

## 🔑 Key Design Decisions

**Why Hybrid ML + LLM?**
The Kaggle Credit Card Fraud dataset uses PCA-transformed features (V1–V28) that encode real behavioural patterns but are not human-interpretable. A Random Forest trained on these features can classify them with high accuracy. The LLM is reserved for borderline cases where human-style reasoning over context adds value — reflecting how real BFSI fraud systems work.

**Why 4 Agents?**
Each agent has a single responsibility matching real bank operations: detection → investigation → routing → notification. This separation makes the system auditable, testable, and extensible.

**Why Balanced Dataset?**
The original dataset is heavily imbalanced (0.17% fraud). Balancing to 50/50 prevents the model from learning to always predict legitimate. The `prepare_data.py` script handles this reproducibly.

---

## 🧪 Tests

```bash
pytest tests/test_agents.py -v
```

---

## 🚀 Live Demo

[Streamlit Cloud Deployment](https://fraud-detection.streamlit.app)

---

## 👩‍💻 Author

**Ramya** — Mainframe professional transitioning to Agentic AI   
Target: Agentic AI Developer roles at BFSI firms and Big Tech