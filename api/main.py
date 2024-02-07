from fastapi import FastAPI, File, UploadFile
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the model file
MODEL_FILE = os.path.join(BASE_DIR, "../saved_models/1")

# # Load the model
MODEL = tf.keras.models.load_model(MODEL_FILE)
# MODEL = tf.keras.models.load_model("../saved_models/1") # I could not get the model with realtive path 

# Class Names
CLASS_NAMES = ['Early_blight', 'Late_blight', 'healthy']

app = FastAPI()


# Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))# Pillow is used ot read images and we would use it to read the bytes as a pillow image and then convert it to numpy array
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)# UploadFile is the datatype(fastapi's) and the default value is set to File(...)
):
    # bytes = await file.read()# We are trying to turn the file to array, we will first turn it to bytes
    image = read_file_image(await file.read())
    img_batch = np.expand_dims(image, 0) #It does not take single image so it take batch image so we increase the dimension. Increasing the dimensions to meake it in the format our model wil want
    predictions = MODEL.predict(img_batch)
    print(predictions[0])
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)


# Loading the model from one specific saved version may work well for demo apps but in big compaines there are new version of the models all the time and we can be using version and the we come up with version 2 we can also say that normal users will use the production model and beta model will be for beta users or testers based on the user type, the best way to do this is using tf serving.
    