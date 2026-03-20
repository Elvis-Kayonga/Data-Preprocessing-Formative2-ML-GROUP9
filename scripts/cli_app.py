import os
import warnings
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

warnings.filterwarnings("ignore", category=UserWarning)

logging.getLogger("tensorflow").setLevel(logging.ERROR)

import joblib
import pandas as pd
import numpy as np
from image_preprocessing import extract_image_features
from voice_preprocessing import process_audio


# Load saved models and encoders
face_model = joblib.load("../models/face_recognition_model.pkl")
voice_model = joblib.load("../models/speaker_model.pkl")
product_model = joblib.load("../models/product_xgb_model.pkl") 
face_encoder = joblib.load("../encoders/face_label_encoder.pkl")
product_label_encoder = joblib.load("../encoders/product_label_encoder.pkl")
product_columns = joblib.load("../encoders/model_columns.pkl")  

# Load merged dataset
merged_dataset = pd.read_csv("../extracted_datasets/merged_dataset.csv")

# Mapping
label_to_customer_id = {
    "sharif": 128,
    "paulette": 103,
    "samuel": 152,
    "kayonga": 121,
}


# Main CLI Function

def main():
    print("\n=== Welcome to Secure Product Recommendation CLI ===\n")

    # Step 1: Face Recognition
    image_path = input("Enter path to face image: ").strip()
    if not os.path.exists(image_path):
        print("Error: Image file not found.")
        return

    image_features = extract_image_features(image_path)
    if image_features is None:
        print("Error processing image.")
        return
    image_features = image_features.reshape(1, -1)

    predicted_face_encoded = face_model.predict(image_features)[0]
    predicted_face = face_encoder.inverse_transform(
        [predicted_face_encoded])[0]

    if predicted_face == "Unknown":
        print("Access Denied: User not recognized.")
        return

    print(f"\nFace recognized as: {predicted_face}")

    # Step 2: Map face to customer ID
    customer_id = label_to_customer_id.get(predicted_face.lower())
    if customer_id is None:
        print("Access Denied: Customer not found in dataset.")
        return

    customer_row = merged_dataset[merged_dataset['customer_id'] == int(
        customer_id)]
    if customer_row.empty:
        print("Error: No data found for this customer.")
        return

    # Step 3: Prepare product features
    product_features = customer_row.reindex(
        columns=product_columns, fill_value=0)
    product_features = product_features.values.astype(float)
    
    # Step 4: Product recommendation
    product_prediction_encoded = product_model.predict(product_features)[0]
    product_prediction = product_label_encoder.inverse_transform(
        [product_prediction_encoded])[0]

    # Step 5: Voice Verification
    audio_path = input("\nEnter path to voice recording: ").strip()
    if not os.path.exists(audio_path):
        print("Error: Audio file not found.")
        return

    audio_features = process_audio(audio_path)
    if audio_features is None:
        print("Error processing audio.")
        return
    audio_features = audio_features.reshape(1, -1) 

    predicted_voice = voice_model.predict(audio_features)[0]

    if predicted_voice.lower() != predicted_face.lower():
        print("Access Denied: Voice does not match recognized face.")
        return

    print(f"\nVoice verified successfully for {predicted_face}.")
    print(f"Recommended Product for {predicted_face} is {product_prediction}\n")

if __name__ == "__main__":
    main()
