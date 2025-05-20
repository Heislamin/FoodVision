# model.py

import tensorflow as tf
import numpy as np
from PIL import Image

# —– Load your Keras model —–
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
) -> str:
    """
    Run inference on a PIL image. Returns:
      - predicted class name ("pizza" or "steak")
    """
    # Prep & batch
    img_tensor = load_and_prep_image(image)
    img_batch  = tf.expand_dims(img_tensor, axis=0)  # shape: (1, H, W, 3)

    # Predict
    preds = model.predict(img_batch)

    if preds.shape[-1] == 1:
        # Binary (sigmoid)
        class_idx = int(tf.round(preds[0][0]).numpy())
    else:
        # Categorical (softmax)
        class_idx = int(np.argmax(preds[0]))

    return CLASS_NAMES[class_idx]
