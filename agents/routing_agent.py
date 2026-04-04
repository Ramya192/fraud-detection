from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config.settings import OPENAI_API_KEY, MODEL_NAME


class RoutingAgent:

    def __init__(self, name):
        self.name = name
        self.routes_made = 0
        self.it_routes = 0
        self.fraud_routes = 0
        self.support_routes = 0
        self.llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            temperature=0
        )

    def route(self, transaction, analyst_response):
        self.routes_made += 1

        prompt = f"""
        You are a bank operations routing system.
        Based on the transaction details and analyst findings,
        route this case to the correct department.

        Transaction:
        - Amount: ${transaction['Amount']}
        - Hour: {transaction['hour']}

        Analyst findings:
        {analyst_response}

        Routing rules:
        - IT Department: network failures, transaction errors,
          system issues, failed transactions
        - Fraud Department: suspicious patterns, card verification
          attacks, unusual amounts, high risk flags
        - General Support: normal customer queries, legitimate
          large purchases, routine account activity

        Respond in exactly this format:
        Department: <IT/FRAUD/GENERAL_SUPPORT>
        Priority: <HIGH/MEDIUM/LOW>
        Reason: <one sentence>
        Action Required: <what the department should do>
        """

        try:
            messages = [
                SystemMessage(content="You are a bank routing system. Route cases to correct departments."),
                HumanMessage(content=prompt)
            ]
            response = self.llm.invoke(messages)

            # Track department counts
            result = response.content
            if "IT" in result.split('\n')[0]:
                self.it_routes += 1
            elif "FRAUD" in result.split('\n')[0]:
                self.fraud_routes += 1
            else:
                self.support_routes += 1

            return result

        except Exception as e:
            print(f"  [RoutingAgent ERROR] {e}")
            self.fraud_routes += 1
            return "Department: FRAUD\nPriority: MEDIUM\nReason: Routing failed — defaulting to Fraud team\nAction Required: Manual review"

    def status(self):
        print(f"{self.name} made {self.routes_made} routing decisions")
        print(f"  → IT Department:      {self.it_routes}")
        print(f"  → Fraud Department:   {self.fraud_routes}")
        print(f"  → General Support:    {self.support_routes}")