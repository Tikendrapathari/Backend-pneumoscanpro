from fastapi import FastAPI, UploadFile, File
from model_loader import load_model_from_drive
from PIL import Image
from io import BytesIO
from enum import Enum
import numpy as np

app = FastAPI()


class Disease(str, Enum):
    covid19="covid19"
    pneumonia="pneumonia"
    tuberculosis="tuberculosis"
    lung_opacity="lung_opacity"
    lung_cancer="lung_cancer"


CLASS_NAMES={

    "covid19":[
        "COVID",
        "Lung Opacity",
        "Normal",
        "Viral Pneumonia"
    ],

    "pneumonia":[
        "Normal",
        "Pneumonia"
    ],

    "tuberculosis":[
        "Normal",
        "Tuberculosis"
    ],

    "lung_opacity":[
        "Normal",
        "Lung Opacity"
    ],

    "lung_cancer":[
        "Benign",
        "Malignant",
        "Normal"
    ]

}


@app.post("/predict")

async def predict(
    disease:Disease,
    file:UploadFile=File(...)
):

    disease=disease.value

    model=load_model_from_drive(
        disease
    )

    content=await file.read()

    img=Image.open(
        BytesIO(content)
    ).convert("RGB")

    img=img.resize((224,224))

    arr=np.array(img)/255.0

    arr=np.expand_dims(arr,0)

    pred=model.predict(arr)

    predicted_index=np.argmax(
        pred
    )

    predicted_class=CLASS_NAMES[disease][
        predicted_index
    ]

    confidence=float(
        np.max(pred)
    )*100

    return {

        "disease":disease,

        "prediction":
        predicted_class,

        "confidence":
        round(confidence,2)

    }