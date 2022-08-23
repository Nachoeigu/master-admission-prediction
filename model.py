import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import re
import joblib


class AdmissionPrediction:

    def __init__(self):
        self.data = pd.read_csv('master_admission.csv')

    def __population_minority_class(self):
        #We create random samples for the minority class with K nearest neighbor technique like SMOTE
        sm = SMOTE()
        x_resampled, y_resampled = sm.fit_resample(self.data.iloc[:,:-1],self.data.iloc[:,-1])
        self.data = x_resampled.join(y_resampled)

    def feature_engineering(self):
        self.__population_minority_class()

        self.ct = ColumnTransformer(
            [
                ('step_1', StandardScaler(), ['gre_score','toefl_score','university_rating','sop','lor','cgpa','research']),
                ('step_2', SimpleImputer(), ['gre_score','toefl_score','university_rating','sop','lor','cgpa','research'])
            ],
            remainder = 'passthrough'
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data.iloc[:,:-1], self.data.iloc[:,-1], test_size = .3, random_state = 2022)

        self.ct.fit(self.X_train)

        self.X_train = self.ct.transform(self.X_train)

    def finding_optimal_algorithm(self):

        models = {
            'SVC': SVC(),
            'LogisticRegression': LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier()    
            }

        params = {
            'RandomForestClassifier':{
                'n_estimators': [100,200,300, 400],
                'criterion': ["gini", "entropy"]
            },
            'SVC':{
                "kernel": ["poly","rbf"],
                "gamma": ['scale','auto']
            },
            'LogisticRegression':{
            },
            
        }
        best_score = 0
        self.best_model = None

        for idx, name in enumerate(models):
            grid_model = GridSearchCV(models[name], params[name], cv = 3)
            grid_model.fit(self.X_train, self.y_train.values.ravel())
            score = accuracy_score(grid_model.predict(self.ct.transform(self.X_test)), self.y_test)

            if score > best_score:
                self.best_model = grid_model.best_estimator_
                best_score = score

        best_model_params = re.sub(".*?\(","(",str(self.best_model)).replace('(','').replace(')','')
        best_model_name = re.sub("\(.*","",str(self.best_model))

        print(f"The best model is {best_model_name} with params {best_model_params}")


    def save_model(self, name):

        joblib.dump(self.best_model, name + '.sav')


    def predict(self, features):
        
        features = self.ct.transform(features)
        
        output =  self.best_model.predict(features)

        if output[0] == 1:
            print("The student is admitted")
        else:
            print("The student is not admitted")
