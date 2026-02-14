from keras.models import load_model
import json
import os

import numpy as np
import pandas as pd
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img

baseDir = os.environ.get(
    "TRAINED_MODEL_DIR",
    os.path.join(os.getcwd(), "trained_model"),
)
databaseDir = os.path.join(os.getcwd(), "data_files")
df = pd.read_csv(open(os.path.join(databaseDir, "supplement_info.csv")))

model = None
data = None


def _ensure_model_loaded():
    global model, data
    if model is not None and data is not None:
        return

    model_path = os.path.join(baseDir, "best_model.h5")
    mapping_path = os.path.join(baseDir, "datafile.json")

    if not os.path.isfile(model_path) or not os.path.isfile(mapping_path):
        raise FileNotFoundError(
            "Missing model files. Expected "
            f"'{model_path}' and '{mapping_path}'."
        )

    print("loading model here")
    model = load_model(model_path)
    print("Model Shape is =>", model.input_shape)
    data = json.load(open(mapping_path))


def prediction(path):
    _ensure_model_loaded()

    img = load_img(path, target_size=(256, 256))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    pred = np.argmax(model.predict(img))
    value = data.get(str(pred))
    if value is None:
        raise ValueError(f"Predicted class index '{pred}' not found in datafile.json")
    print(f" the image belongs to { value } ")
    rows = df.loc[df["disease_name"] == value]
    if rows.empty:
        raise ValueError(f"Predicted label '{value}' not found in supplement_info.csv")
    return rows.values[0][0]


def getDataFromCSV(index):
    if "index" not in df.columns:
        return []
    rows = df.loc[df["index"] == index]
    if rows.empty:
        return []
    return rows.values[0]



if __name__ == "__main__":
    # just for the testing
    path = os.path.join(baseDir, "baseimg.png")
    print(prediction(path))
