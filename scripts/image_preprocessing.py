import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

#load MobileNetV2 for feature extraction 
embedding_model = MobileNetV2(
    weights='imagenet', include_top=False, pooling='avg')

def extract_image_features(image_path):
    """
    Process a new image exactly like training.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image '{image_path}'")

    
    hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    
    img_resized = cv2.resize(img, (224, 224))
    img_array = img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    embedding = embedding_model.predict(img_array, verbose=0).flatten()

    features = np.concatenate([hist, embedding])

    return features
