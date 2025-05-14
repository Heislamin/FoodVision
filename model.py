# model.py
from tensorflow import keras
from PIL import Image
import numpy as np

# —– Load your model —–
# Option A: SavedModel directory
model = keras.models.load_model("mymodel.h5")
# Option B: single HDF5 file
# model = keras.models.load_model("food_classifier.h5")

model.trainable = False

# —– Preprocessing —–
def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    # add batch dim: (1, 224, 224, 3)
    return np.expand_dims(arr, axis=0)

# —– Class names (match your training) —–
CLASS_NAMES = ["pizza", "steak", /* … */]

# —– Prediction function —–
def predict(img: Image.Image) -> str:
    x = preprocess(img)
    probs = model.predict(x)[0]      # e.g. [0.1, 0.7, …]
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx]
