from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OPENAI_API_KEY, MODEL_NAME


class FraudDetectorAgent:

    def __init__(self,name):
        self.name = name
        self.cases_reviewed = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME, 
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def build_prompt(self, transaction, context=None):
        context_str = context if context else "No historical context available"
    
        return f"""
           You are a fraud detection expert at a bank.
    
            Key fraud indicators you must consider:
            - Exactly $0.00 transactions ALWAYS indicate card verification attacks
                — mark as HIGH RISK automatically regardless of hour
            - Amounts between $0.01 and $5.00 at hours 0-5 are suspicious
              (legitimate micro-transactions rarely happen at night)
            - Amounts between $0.01 and $5.00 during daytime (hours 6-22)
              are likely legitimate app subscriptions — do NOT flag automatically
            - Transactions between midnight and 5am (hour 0-5) are HIGH RISK
            - Amounts over $500 at unusual hours are suspicious
            - Legitimate transactions typically have amounts between $10-$500
            - If amount is significantly different from recent average — suspicious
            - Transactions at hour 23 (11pm) with amounts over $200 are suspicious
    
            Historical context:
            {context_str}

            Analyse this transaction and respond with:
            1. Risk Level: LOW / MEDIUM / HIGH
            2. Reason: one sentence explanation
            3. Action: APPROVE / FLAG / BLOCK

            Transaction details:
            - Amount: ${transaction['Amount']}
            - Hour of day: {transaction['hour']} (0=midnight, 23=11pm)
            - Time since first transaction: {transaction['Time']} seconds

            Respond in exactly this format:
            Risk Level: <LOW/MEDIUM/HIGH>
            Reason: <one sentence>
            Action: <APPROVE/FLAG/BLOCK>
        """
    
    def analyse(self, transaction, context=None):
        self.cases_reviewed += 1

        try:
            messages = [
                SystemMessage(content="You are a bank fraud detection expert. Be concise and precise."),
                HumanMessage(content=self.build_prompt(transaction, context))
            ]
            response = self.llm.invoke(messages)
            return response.content

        except Exception as e:
           # Never let one failed API call crash the whole pipeline
           print(f"  [DetectorAgent ERROR] {e}")
           return "Risk Level: MEDIUM\nReason: Analysis unavailable due to API error\nAction: FLAG"
        
    def status(self):
        print(f"{self.name} has reviewed {self.cases_reviewed} transactions")