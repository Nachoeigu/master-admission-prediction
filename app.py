from fastapi import FastAPI
import joblib
import pandas as pd
from model import AdmissionPrediction

app = FastAPI()

@app.get('/')
async def root():
    return {
        'Hi, nice to e-meet you. Would you like to make some predictions about the master admission program?'
    }

@app.get('/predict')
def make_prediction(gre_score:float, toefl_score:float,university_rating:float,sop:float,lor:float,cgpa:float,research:float):
    df = pd.DataFrame(
        [[gre_score,toefl_score,university_rating,sop,lor,cgpa,research]], 
        columns = ['gre_score', 'toefl_score', 'university_rating', 'sop', 'lor', 'cgpa', 'research']
        )
    model = joblib.load("admission.sav")

    ap = AdmissionPrediction()
    ap.feature_engineering()
    ct = ap.ct  
    del ap
      
    if model.predict(ct.transform(df))[0] == 1:
        output  = 'The student is admitted'
    else:
        output  = 'The student is not admitted'

    return {
        'Output': f'{output}'
        }    
