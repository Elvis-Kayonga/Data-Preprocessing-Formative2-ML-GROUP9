# Data Preprocessing Formative 2 (ML GROUP 9)

This repository contains the full assignment workflow for the multimodal project:

1. Task 1: Product recommendation model (transaction + social profile data)
2. Task 2: Face image processing template
3. Task 3: Voice audio processing template
4. Task 4: Multimodal integration demo (face gate -> voice gate -> product model)

All notebooks in `notebooks/` are executed notebooks ready for submission.

## Final Notebook Set (Executed)

- `notebooks/Task1_Product_Recommendation.ipynb`
- `notebooks/Task2_Face_Image_Processing.ipynb`
- `notebooks/Task3_Voice_Audio_Processing.ipynb`
- `notebooks/Task4_Multimodal_Integration.ipynb`

## Project Structure

- `data/raw/`: source CSV files
- `data/processed/`: cleaned and engineered datasets + modality features
- `models/`: trained model and metrics artifacts
- `outputs/plots/`: EDA and model visualizations
- `src/data_pipeline.py`: data cleaning, merge, and EDA plot generation
- `src/train_product_model.py`: model training/evaluation and advanced model plots
- `src/run_all.py`: end-to-end pipeline runner
- `docs/`: teammate guides and submission notes

## How To Run Locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure raw files exist in `data/raw/`:

- `customer_social_profiles.csv`
- `customer_transactions.csv`

3. Run full pipeline:

```bash
python src/run_all.py
```

4. Optional: execute notebooks for verification:

```bash
python -m nbconvert --to notebook --execute notebooks/Task1_Product_Recommendation.ipynb --output Task1_Product_Recommendation.ipynb --output-dir notebooks
python -m nbconvert --to notebook --execute notebooks/Task2_Face_Image_Processing.ipynb --output Task2_Face_Image_Processing.ipynb --output-dir notebooks
python -m nbconvert --to notebook --execute notebooks/Task3_Voice_Audio_Processing.ipynb --output Task3_Voice_Audio_Processing.ipynb --output-dir notebooks
python -m nbconvert --to notebook --execute notebooks/Task4_Multimodal_Integration.ipynb --output Task4_Multimodal_Integration.ipynb --output-dir notebooks
```

## Key Outputs

Data artifacts:
- `data/processed/social_profiles_clean.csv`
- `data/processed/social_profiles_aggregated.csv`
- `data/processed/transactions_clean.csv`
- `data/processed/merged_customer_dataset.csv`
- `data/processed/image_features.csv`
- `data/processed/audio_features.csv`

Model artifacts:
- `models/product_recommendation_model.joblib`
- `models/product_model_metrics.json`
- `models/product_model_comparison.json`

Plots:
- `outputs/plots/01_distributions.png`
- `outputs/plots/02_outliers.png`
- `outputs/plots/03_correlations.png`
- `outputs/plots/04_model_comparison_metrics.png`
- `outputs/plots/05_category_distribution.png`
- `outputs/plots/06_category_purchase_boxplot.png`
- `outputs/plots/07_social_overview.png`
- `outputs/plots/08_confusion_matrix.png`
- `outputs/plots/09_feature_importance.png`
- `outputs/plots/10_face_preview.png`
- `outputs/plots/11_face_augmentation.png`
- `outputs/plots/12_audio_waveform.png`
- `outputs/plots/13_audio_augmentation.png`
- `outputs/plots/14_multimodal_gate_results.png`

## Documentation

See the `docs/` folder for team-friendly guides:

- `docs/ASSIGNMENT_CHECKLIST.md`
- `docs/TEAMMATE_QUICKSTART.md`
- `docs/NOTEBOOK_GUIDE.md`
- `docs/COMMIT_STRATEGY.md`
- `docs/COLAB_SUBMISSION_GUIDE.md`

## Notes

- XGBoost is optional at runtime. If not importable, the pipeline still runs with Random Forest and Logistic Regression.
- Face/audio notebooks handle missing optional dependencies with graceful fallback messages.
- Notebook naming has been standardized to `Task1`..`Task4` for clear assignment mapping.
