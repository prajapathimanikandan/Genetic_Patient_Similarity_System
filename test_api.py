
import pandas as pd
import requests

file_path = r"C:\Users\athir\Downloads\genetic_disease_dataset.csv"

df = pd.read_csv(file_path, encoding="latin1", on_bad_lines="skip")
print("ALL DATASET COLUMNS:")
print(df.columns.tolist())

payload = {
    "records": df.head(500).to_dict(orient="records"),
   "target_column": "Disease",
    "n_components": 2,
    "variance_threshold": 0.95
}
print("Target sent:", payload["target_column"])
print("Column exists:", payload["target_column"] in df.columns)
print("Matching columns:", [repr(c) for c in df.columns if "Symptom" in str(c)])

response = requests.post(
    "http://127.0.0.1:5000/analyze",
    json=payload
)


result = response.json()

print("Status Code:", response.status_code)
print("PCA Variance:", result.get("variance_retained"))
print("LDA Message:", result.get("lda_message"))

if result.get("lda_coordinates") is not None:
    print("LDA is working!")
    print("LDA sample:", result["lda_coordinates"][:5])
else:
    print("LDA is not working yet.")