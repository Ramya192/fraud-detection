import whisper
import tempfile
import os

class VoiceInputTool:

    def __init__(self):
        # load the smallest whisper model , fast and free
        print("Loading Whisper Model...")
        self.model = whisper.load_model ("tiny")
        print("Whisper ready")

    def transcribe(self,audio_file_path):
        """Convert audio file to text"""
        result= self.model.transcribe(audio_file_path)
        return result["text"]
    
    def extract_transaction(self,transcribed_text,llm):
        """Use LLM to extract transaction details from spoken text"""
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = f"""
        Extract transaction details from this spoken description.

        Spoken_text: "{transcribed_text}"

        Return ONLY a JSON object with these exact keys:
        {{
            "Amount:" <number>,
            "hour": <number 0-23>,
            "Time": 50000
        }}

        If amount not mentioned use 0.
        If hour not mentioned use 12.
        Return ONLY the JSON, nothing else.

        """
        messages =[
            SystemMessage(content="Extract transaction data from speech. Return only valid JSON"),
            HumanMessage(content=prompt)
        ]

        response= llm.invoke(messages)
        return response.content