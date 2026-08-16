import os

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., example=40, alias="hours-per-week")
    native_country: str = Field(..., example="United-States", alias="native-country")

# filepath for the enconder and then running the load_model function
path = os.path.join(os.path.dirname(__file__), "model", "encoder.pkl") 
encoder = load_model(path)

# filepath for the model and then running the load_model function
path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
model = load_model(path)

# initialist fast API with title, desc, and version
app = FastAPI(
    title="Census Income Prediction API",
    description="API for making salary category predictions using trained ML model.",
    version="1.0.0"
)


@app.get("/")
async def get_root():
    """ Say hello!"""
    # I didn't see where it stated what the welcome message should say, so I used a generic one
    return {"message": "Hello! Welcome to the Census Income Prediction API!"}


@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data = pd.DataFrame.from_dict(data)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    data_processed, _, _, _ = process_data(
        # these are the inputs defined in data.py
        X=data, # defined as the data frame above
        categorical_features=cat_features, # defined in the list above this code block
        label=None, 
        training=False,
        encoder=encoder, # encoder is defined above using the load_model function to load the encoder.pkl file
        lb=None,
    )
    
    # using the inference function with the data_processed we just created
    _inference = inference(model, data_processed)
    return {"result": apply_label(_inference)}
