from model import AdmissionPrediction 
import pandas as pd

model = AdmissionPrediction()

model.feature_engineering()

model.finding_optimal_algorithm()

model.save("admission")
