import base64
import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import OPENAI_API_KEY

class ImageInputTool:
    
    def __init__(self):
        # Chatgpt mini supports vision
        self.llm = ChatOpenAI(
            model = "gpt-4o-mini",
            api_key = OPENAI_API_KEY,
            temperature=0
        )

    def encode_image(self, image_path):
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    
    def extract_transaction(self, image_path):
        """Extract transaction details from check or receipt image"""

        base64_image = self.encode_image(image_path)

        message = HumanMessage(
            content=[
                {
                    "type":"text",
                    "text":"""Extract transaction details from this cheque or receipt image.
                    Return ONLY a JSON object with these exact keys:
                    {
                        "Amount": <number>,
                        "hour":12,
                        "Time" :50000,
                        "source": "image"
                    }
                    If you cannot read the amount clearly, use 0.
                    Return ONLY the JSON, nothing else."""
                },
                {
                    "type": "image_url",
                    "image_url":{
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                }
            ]
        )

        response=self.llm.invoke([message])
        raw = response.content.strip()
        if "```" in raw:
            raw= raw.split("```")[1]
            if raw.startswith("json"):
                raw= raw[4:]

        return json.loads(raw.strip())
