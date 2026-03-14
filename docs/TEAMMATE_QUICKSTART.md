# Teammate Quickstart

This project has 4 notebooks. If you are a teammate, start here.

## 1) Clone and install

```bash
git clone <repo-url>
cd Data-Preprocessing-Formative2-ML-GROUP9
pip install -r requirements.txt
```

## 2) Run Task 1 notebook (Elvis part)

Open and run:
- `notebooks/Task1_Product_Recommendation.ipynb`

This generates:
- `models/product_recommendation_model.joblib`
- metrics JSON files
- EDA/model plots in `outputs/plots/`

## 3) If you are doing face task

1. Add images in:
- `data/images/<your_name>/`

2. Run:
- `notebooks/Task2_Face_Image_Processing.ipynb`

3. Check output:
- `data/processed/image_features.csv`

## 4) If you are doing voice task

1. Add audio in:
- `data/audio/<your_name>/`

2. Run:
- `notebooks/Task3_Voice_Audio_Processing.ipynb`

3. Check output:
- `data/processed/audio_features.csv`

## 5) Run integration demo

- `notebooks/Task4_Multimodal_Integration.ipynb`

You should see denied/approved scenarios and recommendation output.

## Troubleshooting

- If OpenCV missing:
```bash
pip install opencv-python scikit-image
```

- If librosa missing:
```bash
pip install librosa soundfile
```

- If XGBoost missing:
```bash
pip install xgboost
```

The notebooks already contain fallbacks, so they still run even when optional packages are missing.
