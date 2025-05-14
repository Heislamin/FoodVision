# model.py

import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple

# —– Load your Keras model —–
# Make sure this file sits next to 'food_classifier.h5'
model = tf.keras.models.load_model("food_classifier.h5")
model.trainable = False

# —– Class names —–
CLASS_NAMES = ["pizza", "steak"]

def load_and_prep_image(
    image: Image.Image,
    img_shape: int = 224
) -> tf.Tensor:
    """
    Take a PIL image and:
      1. Resize to (img_shape, img_shape)
      2. Scale pixels to [0,1]
      3. Return a tf.Tensor of shape (img_shape, img_shape, 3)
    """
    # resize & normalize
    image = image.resize((img_shape, img_shape))
    img_array = np.array(image) / 255.0

    # ensure 3 channels
    if img_array.ndim == 2:  # grayscale to RGB
        img_array = np.stack([img_array]*3, axis=-1)
    return tf.convert_to_tensor(img_array, dtype=tf.float32)

def predict_image(
    image: Image.Image
) -> Tuple[str, float]:
    """
    Run inference on a PIL image. Returns:
      - predicted class name ("pizza" or "steak")
      - confidence score (0.0–1.0 for the predicted class)
    """
    # Prep & batch
    img_tensor = load_and_prep_image(image)
    img_batch  = tf.expand_dims(img_tensor, axis=0)  # shape: (1, H, W, 3)

    # Predict: returns array of shape (1,1) with values in [0,1] if using sigmoid,
    # or (1,2) probabilities if using softmax—adjust below accordingly.
    preds = model.predict(img_batch)

    # Handle both sigmoid (binary) and softmax (2-class) outputs:
    if preds.shape[-1] == 1:
        # sigmoid: preds = [[0.73]] → round to 0 or 1
        prob   = float(preds[0][0])
        class_idx = int(tf.round(prob).numpy())
    else:
        # softmax: preds = [[0.1, 0.9]]
        prob      = float(np.max(preds[0]))
        class_idx = int(np.argmax(preds[0]))

    class_name = CLASS_NAMES[class_idx]
    return class_name, prob
