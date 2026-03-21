# Entry point for Fraud Detection Multi-Agent System
# Built by Ramya - 2026
# Session 2 complete — classes, pip, .env, imports done
# Day 1 complete — data explored, cleaned, balanced (984 txns)
# Next: Day 2 — connect agent to real transaction data

import pandas as pd
from agents.detector_agent import FraudDetectorAgent
from agents.analyst_agent import AnalystAgent
from agents.alert_agent import AlertAgent
from config.settings import RISK_THRESHOLD, MODEL_NAME
from tools.image_input import ImageInputTool
import json

print("Model:", MODEL_NAME)
print("Risk Threshold:", RISK_THRESHOLD)

# ----------CSV  INPUT DEMO ----------------------------------------------
# Load Balanced Dataset
df = pd.read_csv("data/transactions_balanced.csv")

# Pick 5 Fraud + 5 Legitimate transactions
fraud_sample = df[df["is_Fraud"] ==1].head(25)
legit_sample = df[df["is_Fraud"] ==0].head(25)
test_df = pd.concat([fraud_sample, legit_sample]).reset_index(drop=True)

# Create Agent
agent = FraudDetectorAgent(name="DetectorAgent-1")
analyst = AnalystAgent(name="AnalystAgent-1")
alert_agent = AlertAgent(name="AlertAgent-1")

avg_amount = df["Amount"].mean()
max_amount = df["Amount"].max()
context = f"Dataset average transaction amount: ${avg_amount:.2f}. Maximum amount: ${max_amount:.2f}. Typical legitimate transactions range from $5 to $500."

# Track Results
correct = 0
wrong = 0
results = []

print("=" * 60)
print("FRAUD DETECTION AGENT - 10 TRANSACTIONS TEST")
print("=" * 60)

for index, row in test_df.iterrows():
    transaction = row.to_dict()
    actual = "FRAUD" if transaction["is_Fraud"] == 1 else "LEGITIMATE"


    # get agent response
    response = agent.analyse(transaction, context)

    # Check if Agent said HIGH RISK
    if "HIGH" in response:
        predicted = "FRAUD"
    else:
        predicted = "LEGITIMATE"

    # Track Accuracy
    is_correct = predicted == actual
    if is_correct:
        correct += 1
    else:
        wrong += 1

    # store Result
    results.append({
        "amount": transaction["Amount"],
        "hour": transaction["hour"],
        "actual": actual,
        "predicted": predicted,
        "correct": is_correct,
    })

    # print Result
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    print(f"Txn {index +1}: $ {transaction["Amount"]:>8.2f} |"
          f"Hour: {int(transaction["hour"]):>2} |"
          f"Actual: {actual:<12} | "
          f"Predicted: {predicted:<12} | {status}")
    
    if predicted == "FRAUD":
        print(f"\n >>> ESCALATING TO ANALYST AGENT...")
        investigation = analyst.investigate(transaction, response)
        print("Investigation Report:")
        for line in investigation.split("\n"):
            if line.strip():
                print(f"  {line}")
            print()
        
        # If analyst recommends BLOCK - generate formal alert
        if "BLOCK" in investigation:
            print(f"  >>> GENERATING FORMAL ALERT...")
            alert= alert_agent.generate_alert(transaction, response, investigation)

            print(f" FORMAL ALERT:")
            print(" " + "=" * 50)
            for line in alert.split("\n"):
                if line.strip():
                    print(f"  {line}")
            print(" " + "=" * 50)
            print()           


print("=" * 60)
total = len(test_df)
accuracy = round((correct / total) * 100, 1)
print(f"Results: {correct}/{total} correct ({accuracy}% accuracy)")
print(f"Wrong: {wrong}/{total}")
print("=" *60)

# Detailed Metrics
fraud_results = [r for r in results if r["actual"]== "FRAUD"]
legit_results = [r for r in results if r["actual"]== "LEGITIMATE"]

true_positives = sum(1 for r in fraud_results if r["predicted"] == "FRAUD")
false_negatives = sum(1 for r in fraud_results if r["predicted"] == "LEGITIMATE")
true_negatives = sum(1 for r in legit_results if r["predicted"] == "LEGITIMATE")
false_positives = sum(1 for r in legit_results if r["predicted"] == "FRAUD")

print("\nDETAILED METRICS:")
print(f"True Positives  (fraud caught):        {true_positives}")
print(f"False Negatives (fraud missed):        {false_negatives}")
print(f"True Negatives  (legit approved):      {true_negatives}")
print(f"False Positives (legit wrongly blocked): {false_positives}")
print()

# Precision and Recall
if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives)
    print(f"Precision: {round(precision * 100, 1)}%")

if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
    print(f"Recall:    {round(recall * 100, 1)}%")


# print("\n" + "=" * 60)
# print("VOICE INPUT DEMO")
# print("=" *60)

# # ----------VOICE  INPUT DEMO ----------------------------------------------

# # Simulate what whisper could transcribe from audio
# # In production this would come from a real audio file
# spoken_text = "I need to flag a transaction of 850 dollars at 2 am"

# print(f"Spoken: '{spoken_text}'")
# print("Extracting transaction details...")

# # Use LLM to extract transaction from spoken text
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from config.settings import OPENAI_API_KEY, MODEL_NAME

# llm = ChatOpenAI(model = MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)

# extract_prompt = f"""
# Extract transaction details from this spoken description.
# Spoken_text:"{spoken_text}"

# Return ONLY a JSON object with these exact keys:
# {{
#     "Amount": <number>,
#     "hour": <number 0-23>,
#     "Time": 50000
# }}

# Return ONLY the JSON, nothing else.
# """

# messages = [
#     SystemMessage(content="Extract transaction data. Return only valid JSON."),
#     HumanMessage(content=extract_prompt)
# ]

# response = llm.invoke(messages)
# raw_json = response.content.strip()

# # Clean JSON if wrapped in code blocks
# if "```" in raw_json:
#     raw_json = raw_json.split("```")[1]
#     if raw_json.startswith("json"):
#         raw_json = raw_json[4:]

# voice_transaction = json.loads(raw_json)
# print(f"Extracted: Amount=${voice_transaction['Amount']}, Hour={voice_transaction['hour']}")
# print()

# # Run through full 3-agent pipeline
# print("Running through fraud detection pipeline...")
# voice_result = agent.analyse(voice_transaction, context)
# print("Detector:", voice_result.split('\n')[0])

# if "HIGH" in voice_result:
#     voice_investigation = analyst.investigate(voice_transaction, voice_result)
#     print("Analyst:", voice_investigation.split('\n')[2])
    
#     if "BLOCK" in voice_investigation:
#         voice_alert = alert_agent.generate_alert(
#             voice_transaction,
#             voice_result,
#             voice_investigation
#         )
#         print("\nFORMAL ALERT GENERATED:")
#         print("=" * 40)
#         for line in voice_alert.split('\n'):
#             if line.strip():
#                 print(line)
#         print("=" * 40)

# ----------IMAGE INPUT DEMO ----------------------------------------------
# print("\n" +  "=" * 60)
# print("IMAGE INPUT DEMO")
# print("=" * 60)

# image_tool = ImageInputTool()
# image_path ="data/test_images/test_receipt.jpg"

# print(f"Reading Receipt image:{image_path}")
# print("Extracting transaction details using GPT-4V...")

# try:
#     image_transaction = image_tool.extract_transaction(image_path)
#     print(f"Extracted Amount= ${image_transaction['Amount']},"
#           f"Hour={image_transaction['hour']}")
#     print()

#     # Run through full pipeline
#     print("Running through Fraud Detection Pipeline...")
#     image_result = agent.analyse(image_transaction, context)
#     print("Detector:", image_result.split('\n')[0])

#     if "HIGH" in image_result:
#         image_investigation = analyst.investigate(image_transaction,image_result)

#         print("Analyst:", image_investigation.split('\n')[2])

#     if "BLOCK" in image_investigation:
#         image_alert = alert_agent.generate_alert(image_transaction,image_result,image_investigation)

#         print("\nFORMAL ALERT GENERATED")
#         print("=" * 40)
#         for line in image_alert.split('\n'):
#             if line.strip():
#                 print(line)
#         print("=" * 40)

# except Exception as e:
#     print(f"Image processing error: {e}")


agent.status()
analyst.status()
alert_agent.status()