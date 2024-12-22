from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# Load the trained model
model_path = 'final_model_joblib.pkl'  # Path to your saved model
clf = load(model_path)

# Initialize FastAPI app
app = FastAPI()

# Define a request body schema
class PredictionInput(BaseModel):
    age: int
    gender: int
    education: int
    income: float
    loan_amount: float
    loan_intent: int
    credit_score: float

# Define a response model (optional but recommended)
class PredictionOutput(BaseModel):
    prediction: int
    probabilities: dict

# Define the prediction endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    # Convert input data to a NumPy array
    input_array = np.array([[data.age, data.gender, data.education, data.income, 
                             data.loan_amount, data.loan_intent, data.credit_score]])
    
    # Make predictions
    prediction = clf.predict(input_array)[0]  # Single value
    prediction_probabilities = clf.predict_proba(input_array)[0].tolist()  # Convert to list for JSON serialization
    
    # Prepare response
    probabilities = {f"class_{i}": prob for i, prob in enumerate(prediction_probabilities)}
    return PredictionOutput(prediction=int(prediction), probabilities=probabilities)
