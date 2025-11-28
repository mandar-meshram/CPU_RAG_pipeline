from typing import Optional
import requests
from app.config import settings


class LLMClient:

    def __init__(self, api_url : Optional[str] = None):
        self.api_url = api_url or settings.LLM_API_URL

    def generate(self, prompt : str, max_tokens : int = 256, temperature : float = 0.7) -> str:
        payload = {
            "prompt" : prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "stop": ["\n", "Question:", "Context:", "Source:"],
        }
        response = requests.post(self.api_url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "content" in data:
            return data["content"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        return str(data)