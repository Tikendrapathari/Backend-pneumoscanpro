import os
import gdown
import tensorflow as tf

MODELS={

    "pneumonia":{
        "drive_id":"1VZ0SYbf9Vpb1JWivZu8UP_RxifXltJxs",
        "path":"models/pneumonia_model.h5"
    },

    "covid19":{
        "drive_id":"1ZiquTMN2tUnTpe_IpU4KAA78W1MsQXED",
        "path":"models/covid19_model_best.h5"
    },

    "tuberculosis":{
        "drive_id":"1mm3BaWiR5GN9183720Xe8OlEK_MEPzzQ",
        "path":"models/tuberculosis_model_best.h5"
    },

    "lung_opacity":{
        "drive_id":"13b1IE-N15GaCh7D6SWqDUIzkSXDtLuGH",
        "path":"models/lung_opacity_model.h5"
    },

    "lung_cancer":{
        "drive_id":"16pulsOkbYw4fNJx65hpHHH0nTgxNGdMW",
        "path":"models/lung_cancer_model_3class.keras"
    }

}

loaded_models={}


def load_model_from_drive(model_name):

    if model_name in loaded_models:
        return loaded_models[model_name]

    info=MODELS[model_name]

    file_path=info["path"]

    if not os.path.exists(file_path):

        os.makedirs("models",exist_ok=True)

        file_id=info["drive_id"]

        url=f"https://drive.google.com/uc?id={file_id}"

        print("Downloading:",model_name)

        gdown.download(
            url,
            file_path,
            quiet=False
        )

    print("Loading:",model_name)

    model=tf.keras.models.load_model(
        file_path,
        compile=False
    )

    loaded_models[model_name]=model

    return model