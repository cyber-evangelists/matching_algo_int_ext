from fastapi import FastAPI, HTTPException
from model import Questionnaire
import pandas as pd
import numpy as np
import pickle

pd.set_option("display.max_columns", None)

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

app = FastAPI()


# Assuming `random_forest_model` and `scaler` are your trained model and scaler, loaded from disk


@app.post("/predict")
def predict(answers: Questionnaire):
    try:
        scaler, random_forest_model = upload_model()
        answer_data = {}
        if len(answers.answers) == 91:
            for i in range(len(answers.answers)):
                answer_data[f"Q{i+1}A"] = answers.answers[i]
                answer_data[f"Q{i+1}E"] = answers.estimated_time[i]
            answer_data["gender"] = answers.gender
            answer_data["engnat"] = answers.engnat
            answer_data["age"] = answers.age
            print(answer_data)
            answer_data = pd.DataFrame(answer_data, index=[0])
            # answers =answers.replace(np.nan,3, regex=True)
            answer_data = pd.DataFrame(
                scaler.transform(answer_data),
                index=answer_data.index,
                columns=answer_data.columns,
            )

            result = random_forest_model.predict(answer_data)
            return {"prediction": result.item()}

        raise HTTPException(status_code=404, detail="No all questions answer received")
    except Exception as Error:
        raise HTTPException(status_code=404, detail=f"{Error}")


def upload_model():
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    with open("random_forest_model.pkl", "rb") as file:
        random_forest_model = pickle.load(file)
    return scaler, random_forest_model
