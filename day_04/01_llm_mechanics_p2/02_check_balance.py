import os
import requests
from dotenv import load_dotenv

load_dotenv()

data = requests.get("https://openrouter.ai/api/v1/credits",
                    headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                    timeout=10).json()["data"]

print(f"balance: ${data['total_credits'] - data['total_usage']:.4f})")