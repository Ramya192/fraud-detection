# agents/detector_agent.py
# Hybrid DetectorAgent: Rule-based → ML Scorer → LLM (borderline only)
#
# Decision flow:
#   1. Rule-based pre-filter  → catches obvious fraud instantly (no API call)
#   2. ML Scorer (RF)         → fast-path for clear fraud/legit (no API call)
#   3. LLM                    → only called for borderline cases (0.3 – 0.7)

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OPENAI_API_KEY, MODEL_NAME
from tools.ml_scorer import MLScorer


class FraudDetectorAgent:

    def __init__(self, name):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )
        # ML scorer trains on startup — once, not per transaction
        print("  [DetectorAgent] Loading ML scorer...")
        self.scorer = MLScorer(data_path="data/transactions_balanced.csv")

    # ── LAYER 1: Rule-based pre-filter ──────────────────────────────────
    # Catches obvious fraud BEFORE calling ML or LLM.
    # Returns a ready-made response string if a rule fires, else None.
    def rule_based_filter(self, transaction):
        amount = transaction.get("Amount", 0)
        hour   = transaction.get("hour", 12)

        # Rule 1: Exactly $0.00 — card verification attack
        if amount == 0.0:
            return (
                "Risk Level: HIGH\n"
                "Reason: Zero-dollar transaction indicates a card verification attack.\n"
                "Action: BLOCK"
            )

        # Rule 2: Micro-transaction at night — card testing
        if 0 < amount <= 5.0 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Micro-transaction during night hours — classic card-testing pattern.\n"
                "Action: BLOCK"
            )

        # Rule 3: Very high amount at night
        if amount > 1000 and 0 <= hour <= 5:
            return (
                "Risk Level: HIGH\n"
                "Reason: Large transaction amount during overnight hours is highly suspicious.\n"
                "Action: BLOCK"
            )

        return None   # no rule fired

    # ── LAYER 2: ML fast-path ────────────────────────────────────────────
    # Returns a ready-made response string for clear cases, else None.
    def ml_filter(self, transaction):
        score = self.scorer.score(transaction)

        if score >= MLScorer.FRAUD_THRESHOLD:
            return (
                f"Risk Level: HIGH\n"
                f"Reason: ML model fraud probability {score:.0%} — strongly matches fraud patterns in training data.\n"
                f"Action: BLOCK"
            )

        if score <= MLScorer.LEGIT_THRESHOLD:
            return (
                f"Risk Level: LOW\n"
                f"Reason: ML model fraud probability {score:.0%} — strongly matches legitimate transaction patterns.\n"
                f"Action: APPROVE"
            )

        # Borderline — return None so LLM takes over
        return None

    # ── LAYER 3: LLM prompt (borderline cases only) ──────────────────────
    def build_prompt(self, transaction, context=None, ml_score=None):
        context_str = context if context else "No historical context available"

        # Include ML score as a hint when available
        ml_hint = ""
        if ml_score is not None:
            ml_hint = f"""
            ML Model Assessment:
            - Fraud probability score: {ml_score:.0%}
            - This score is BORDERLINE (between 30% and 70%) — human reasoning needed.
            - Weight this alongside the transaction features below.
            """

        return f"""
            You are a fraud detection expert at a bank.
            A machine learning model has flagged this transaction as borderline.
            Your job is to make the final call using your expert reasoning.

            Key fraud indicators:
            - Exactly $0.00 transactions ALWAYS indicate card verification attacks
            - Amounts between $0.01 and $5.00 at hours 0-5 are suspicious
            - Transactions between midnight and 5am (hour 0-5) are HIGH RISK
            - Amounts over $500 at unusual hours are suspicious
            - Legitimate transactions typically have amounts between $10-$500
            - Transactions at hour 23 (11pm) with amounts over $200 are suspicious

            Historical context:
            {context_str}
            {ml_hint}

            Transaction details:
            - Amount: ${transaction['Amount']}
            - Hour of day: {transaction['hour']} (0=midnight, 23=11pm)
            - Time since first transaction: {transaction['Time']} seconds

            Respond in exactly this format:
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """

    # ── Main entry point ─────────────────────────────────────────────────
    def analyse(self, transaction, context=None):
        self.cases_reviewed += 1

        # Layer 1: Rules
        rule_result = self.rule_based_filter(transaction)
        if rule_result:
            return rule_result

        # Layer 2: ML scorer
        ml_result = self.ml_filter(transaction)
        if ml_result:
            return ml_result

        # Layer 3: LLM — only for borderline cases
        ml_score = self.scorer.score(transaction)
        try:
            messages = [
                SystemMessage(content="You are a bank fraud detection expert. Be concise and precise."),
                HumanMessage(content=self.build_prompt(transaction, context, ml_score))
            ]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
            print(f"  [DetectorAgent ERROR] {e}")
            return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"

    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} transactions")
