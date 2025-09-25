from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from src.features import build_feature_row
from src.models import Forecaster

app = FastAPI()
forecaster = Forecaster()

class Frame(BaseModel):
    time_stamp: str
    temperature_active1: float
    temperature_active2: float
    temperature_active3: float
    temperature_active4: float
    temperature_curing1: float
    temperature_curing2: float
    oxygen: float | None = None
    co2: float | None = None
    methane: float | None = None
    # add moisture/pH/etc. fields if your features use them

@app.post("/predict")
def predict(frames: list[Frame]):
    df = pd.DataFrame([f.dict() for f in frames])
    feats = build_feature_row(df)
    y = forecaster.model.predict(feats.values)[0].tolist()
    return {"forecast": y}
