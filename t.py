import pandas as pd

df_chunks = pd.read_json("chunks_marseille_cache.json")

print("Colonnes disponibles :")
print(df_chunks.columns.tolist())

print("\nPremier chunk :")
print(df_chunks.iloc[0])
