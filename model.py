# model.py

import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple

# Load trained Keras model
model = tf.keras.models.load_model("food_classifier.h5")
model.trainable = False

# Class names — must match model training order
CLASS_NAMES = ["pizza", "steak"]

def load_and_prep_image(image: Image.Image, img_shape: int = 224, normalize: bool = True) -> tf.Tensor:
    """
    Resize, convert to RGB, normalize, and convert image to tensor.
    """
    image = image.convert("RGB")
    image = image.resize((img_shape, img_shape))
    image_array = np.array(image)

    if normalize:
        image_array = image_array / 255.0

    if image_array.ndim == 2:
        image_array = np.stack([image_array] * 3, axis=-1)

    return tf.convert_to_tensor(image_array, dtype=tf.float32)

def predict_image(image: Image.Image) -> str:
    """
    Predicts class name only (e.g., 'pizza' or 'steak'), no percentage returned.
    """
    img_tensor = load_and_prep_image(image)
    img_batch = tf.expand_dims(img_tensor, axis=0)

    preds = model.predict(img_batch)

    if preds.shape[-1] == 1:
        class_idx = int(tf.round(preds[0][0]).numpy())
    else:
        class_idx = int(np.argmax(preds[0]))

    return f"It’s a {CLASS_NAMES[class_idx]}."
