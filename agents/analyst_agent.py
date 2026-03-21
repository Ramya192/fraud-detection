from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OPENAI_API_KEY, MODEL_NAME

class AnalystAgent:
    def __init__(self,name):
        self.name= name
        self.investigations = 0
        self.llm = ChatOpenAI(
            model = MODEL_NAME,
            api_key = OPENAI_API_KEY,
            temperature = 0
        )

    def investigate(self, transaction,detector_response):
        self.investigations += 1

        try:
            prompt = f"""
                You are a senior fraud analyst at a bank.
                The fraud detector has flagged  this transaction as HIGH RISK.property
 
                Your job is to provide a detailed investigation report.property

                Flagged transaction:
                - Amount: ${transaction["Amount"]}
                - Hour of day: {transaction["hour"]} (0=midnight, 23=11pm)
                - Time in dataset: {transaction["Time"]} seconds 

                Detector's initial assessment:
                {detector_response}

                Provide a structured investigation report with:
                1. Fraud Pattern: <type>
                2. Risk Factors: List exactly 3 risk factors present
                3. Recommended Action: BLOCK / ESCALATE / MONITOR
                4. Condfidence: LOW/ MEDIUM / HIGH
                5. Next Steps: one sentence on what bank should do

                Respond in exactly this format:
                Fraud Pattern: <type>
                Risk Factors: <factor1>, <factor2>, <factor3>
                Recommended Action: BLOCK / ESCALATE / MONITOR
                Confidence: LOW/ MEDIUM / HIGH
                Next Steps: <one sentence>
                """

            messages = [
                SystemMessage(content="You are a senior bank fraud analyst. Be precise and professional."),
                HumanMessage(content=prompt)
            ]
    
            response = self.llm.invoke(messages)
            return response.content
        
        except Exception as e:
            print(f"  [AnalystAgent ERROR] {e}")
            return "Fraud Pattern: Unknown\nRisk Factors: Analysis unavailable\nRecommended Action: ESCALATE\nConfidence: LOW\nNext Steps: Manual review required"

    def status(self):
      print(f"{self.name} has investigated {self.investigations} cases")