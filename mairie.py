from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os

MISTRAL_API_KEY = "mstrl_xCUvqgerPKqs1Egg5z62jgwCwKW3AOTp_2KoWMp"

# Initialisation du client avec votre clé API
api_key = os.environ["MISTRAL_API_KEY"]
client = MistralClient(api_key=api_key)


def embed_text(text):
    # Appel de l'API d'embedding
    embeddings_batch = client.embeddings(model="mistral-embed", input=text)
    return embeddings_batch.data[0].embedding
