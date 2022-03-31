import requests
import json
import pandas as pd

df = pd.read_csv("preprocessed_data.csv")

df=df[:1].to_json()

r = requests.post("http://127.0.0.1:8000/predict", data=json.dumps(df))

print (r.json())