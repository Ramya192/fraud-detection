# Agentic Fraud Detection System

A production-style multi-agent AI system for detecting fraudulent 
bank transactions using LLMs, prompt engineering, and agentic workflows.

Built by Ramya | Project A

## Live Demo
[Click here to try the live app](#) ← update after deployment

## Problem Statement
Credit card fraud costs banks billions annually. Traditional 
rule-based systems miss sophisticated fraud patterns. This system 
uses 3 specialised AI agents working in a triage pipeline to detect, 
investigate, and alert on fraudulent transactions.

## Architecture
```
Transaction Input (CSV / Voice / Image)
           ↓
   DetectorAgent
   Fast screening — HIGH/MEDIUM/LOW risk
           ↓ (if HIGH)
   AnalystAgent  
   Deep investigation — fraud pattern + risk factors
           ↓ (if BLOCK recommended)
   AlertAgent
   Formal bank notification with severity + actions
           ↓
   Bank Officer — reviews structured alert
```

## Key Features
- 3-agent triage pipeline (Detector → Analyst → Alert)
- Multi-modal inputs: CSV upload, voice, receipt image
- Real-time precision/recall metrics dashboard
- Prompt engineered through 3 iterations
- Production-grade error handling on all agent calls
- Downloadable results as CSV

## Performance Metrics
| Metric | Value |
|--------|-------|
| Precision | 88.9% |
| Recall | 32.0% |
| Dataset | 284,807 transactions (Kaggle) |
| Fraud rate | 0.17% |

## Tech Stack
- Python 3.13
- LangChain + LangChain-OpenAI
- OpenAI GPT-4o-mini
- Streamlit
- Pandas
- Pytest

## Project Structure
```
fraud_detection/
├── agents/
│   ├── detector_agent.py   # screens all transactions
│   ├── analyst_agent.py    # investigates flagged ones
│   └── alert_agent.py      # generates formal alerts
├── tools/
│   ├── voice_input.py      # Whisper voice processing
│   └── image_input.py      # GPT-4V receipt reading
├── data/
│   └── transactions_balanced.csv
├── tests/
│   └── test_agents.py      # 6 pytest tests
├── config/
│   └── settings.py
├── app.py                  # Streamlit dashboard
├── main.py                 # batch pipeline
└── requirements.txt
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Ramya192/fraud-detection.git
cd fraud-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 4. Download dataset
Download `creditcard.csv` from 
[Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place in `data/` folder.

### 5. Run the dashboard
```bash
streamlit run app.py
```

### 6. Run tests
```bash
pytest tests/ -v
```

## What I Learned
- Multi-agent system design using triage pattern
- Prompt engineering through iterative refinement
- Precision-recall tradeoff in imbalanced datasets
- Production-grade error handling for AI pipelines
- Multi-modal AI: text, voice, and image processing