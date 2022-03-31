from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
import pandas as pd
import pickle


# Instantiate the app.
app = FastAPI()

class Inference_model(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="ours-per-week")
    native_country: str = Field(alias="native-country")
    salary: str

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to the project 'scalable pipeline'"}

pkl_filename = "./starter/model.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

# Defining the prediction endpoint without data validation
@app.post('/predict')
async def predict(inference: Inference_model):
	
	# Converting input data into Pandas DataFrame
	input_df = pd.DataFrame([inference.dict()])
	
	# Getting the prediction from the Logistic Regression model
	pred = model.predict(input_df)[0]
	
	return 


