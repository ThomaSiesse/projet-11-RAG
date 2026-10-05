from mistralai.client import MistralClient
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=api_key)

# Test simple : une question basique
try:
    response = client.chat.complete(
        model="mistral-small",
        messages=[{"role": "user", "content": "Bonjour, es-tu Mistral ?"}],
    )
    print("✅ Mistral chat fonctionne !")
    print(f"Réponse : {response.choices[0].message.content}")

except Exception as e:
    print(f"❌ Erreur Mistral chat : {e}")
