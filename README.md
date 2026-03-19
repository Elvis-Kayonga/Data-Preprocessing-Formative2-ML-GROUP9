# MultiModal Data Preprocessing Formative 2 (ML GROUP 9)

This repository contains the current multimodal assignment assets for:

1. Product recommendation using transaction and social profile data
2. Face feature extraction and face recognition model training
3. Voice feature extraction and speaker recognition model training

## Current Project Structure

- encoders
  - face_label_encoder.pkl
  - model_columns.pkl
  - product_label_encoder.pkl
- extracted_datasets/
  - audio_features.csv
  - images.zip
  -merged_dataset.csv
- models/
  - face_recognition_model.pkl
  - face_label_encoder (1).pkl
  - speaker_model.pkl
- notebooks/
  - product_Recommendation_model.ipynb
  - FaceRecognition.ipynb
  - voice_recognition_model.ipynb
- scripts/
  - cli_app.py
  - image_preprocessing.py
  - voice_preprocessing.py

## Notebooks

- notebooks/product_Recommendation_model.ipynb
  - Cleans and merges tabular data
  - Performs  model training for product recommendation
  - Uses Xgb algorithm and saves it and the encoder file
- notebooks/FaceRecognition.ipynb
  - Loads images from images.zip
  - Extracts image features (histogram + MobileNetV2 embeddings)
  - Trains and saves a face recognition model and encoder file
- notebooks/voice_recognition_model.ipynb
  - Loads voice samples
  - Extracts MFCC-based audio features
  - Trains and saves a speaker recognition model 

## Scripts

- scripts/image_preprocessing.py
  - Inference helper for extracting  image feature vectors
- scripts/voice_preprocessing.py
  - Inference helper for extracting 15-dimensional audio feature vectors
- cli_app.py
  -main script to run the authentication process using voice and image models to give access to recommended product frpm recommendation model

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Notes

- The voice notebook currently includes Google Colab-specific paths and drive mounting.
- The face notebook trains and saves a label encoder named face_label_encoder.pkl
