# Teammate Quickstart (Collaboration + Branch Workflow)

This guide shows exactly how teammates should work on separate branches and safely combine work for final submission.

## Goal

By the end, each teammate should be able to:

1. Pull the latest project state.
2. Work only in their own branch.
3. Run only the notebook(s) for their task.
4. Commit cleanly with meaningful messages.
5. Push their branch to GitHub.
6. Open a Pull Request to `elvis` (or `main` if instructed).

## Repository and Branching Model

Recommended branch strategy:

1. `main` = stable baseline.
2. `elvis` = integration branch for this assignment delivery.
3. Teammate branches = personal feature branches (for example `alice-face`, `bob-voice`, `charles-integration`).

Rule:

1. Never commit directly on `main`.
2. Prefer committing work on your own branch, then submit PR.

## One-time Setup (Per Teammate)

## 1) Clone repo

```bash
git clone https://github.com/Elvis-Kayonga/Data-Preprocessing-Formative2-ML-GROUP9.git
cd Data-Preprocessing-Formative2-ML-GROUP9
```

## 2) Install dependencies

```bash
pip install -r requirements.txt
```

## 3) Confirm remote branches

```bash
git fetch --all
git branch -a
```

## Daily Start Workflow (Always Do This First)

Before doing any work:

```bash
git fetch origin
git checkout elvis
git pull origin elvis
```

Create/update your own branch from latest `elvis`:

```bash
git checkout -B <your-branch-name>
```

Examples:

1. `git checkout -B alice-face`
2. `git checkout -B bob-voice`
3. `git checkout -B diana-integration`

## Task Mapping (Who Does What)

1. Task 1 Product model notebook: [notebooks/Task1_Product_Recommendation.ipynb](notebooks/Task1_Product_Recommendation.ipynb)
2. Task 2 Face notebook: [notebooks/Task2_Face_Image_Processing.ipynb](notebooks/Task2_Face_Image_Processing.ipynb)
3. Task 3 Voice notebook: [notebooks/Task3_Voice_Audio_Processing.ipynb](notebooks/Task3_Voice_Audio_Processing.ipynb)
4. Task 4 Integration notebook: [notebooks/Task4_Multimodal_Integration.ipynb](notebooks/Task4_Multimodal_Integration.ipynb)

## Task-specific Steps

## A) Face teammate

1. Add image files to `data/images/<your_name>/`.
2. Run [notebooks/Task2_Face_Image_Processing.ipynb](notebooks/Task2_Face_Image_Processing.ipynb).
3. Verify output file exists: `data/processed/image_features.csv`.
4. If the notebook says OpenCV missing, install and rerun:

```bash
pip install opencv-python scikit-image
```

## B) Voice teammate

1. Add `.wav` files to `data/audio/<your_name>/`.
2. Run [notebooks/Task3_Voice_Audio_Processing.ipynb](notebooks/Task3_Voice_Audio_Processing.ipynb).
3. Verify output file exists: `data/processed/audio_features.csv`.
4. If the notebook says librosa missing, install and rerun:

```bash
pip install librosa soundfile
```

## C) Integration teammate

1. Pull latest branch updates first.
2. Run [notebooks/Task4_Multimodal_Integration.ipynb](notebooks/Task4_Multimodal_Integration.ipynb).
3. Confirm gate simulation output and chart generation.

## Commit and Push Workflow (Every Teammate)

After finishing your task:

## 1) Check what changed

```bash
git status
```

## 2) Stage only relevant files

Examples:

```bash
git add notebooks/Task2_Face_Image_Processing.ipynb
git add data/processed/image_features.csv
git add outputs/plots/10_face_preview.png outputs/plots/11_face_augmentation.png
```

## 3) Commit with clear message

```bash
git commit -m "feat(face): add executed notebook and extracted image features"
```

## 4) Push your branch

```bash
git push -u origin <your-branch-name>
```

## 5) Open Pull Request

On GitHub:

1. Open PR from your branch -> `elvis`.
2. Add a short PR description:
	- what you changed
	- which notebook was executed
	- which outputs were generated
3. Request review.

## If Push Is Rejected

If you see a non-fast-forward error:

```bash
git fetch origin
git pull --rebase origin <your-branch-name>
git push origin <your-branch-name>
```

If conflicts occur during rebase:

1. Edit conflicted files.
2. `git add <file>`
3. `git rebase --continue`
4. Push again.

## Integration Maintainer Flow (Elvis Branch Owner)

When reviewing teammate PRs:

1. Pull latest `elvis`.
2. Merge one PR at a time.
3. Run quick verification after each merge:

```bash
python src/run_all.py
python -m nbconvert --to notebook --execute notebooks/Task4_Multimodal_Integration.ipynb --output Task4_Multimodal_Integration.ipynb --output-dir notebooks
```

4. If everything passes, push updated `elvis`.

## Final Submission Checklist

Before final submission:

1. All 4 notebooks exist and are executed.
2. No obsolete notebook name variants remain.
3. Key artifacts exist:
	- `models/product_recommendation_model.joblib`
	- `models/product_model_metrics.json`
	- `data/processed/merged_customer_dataset.csv`
	- `data/processed/image_features.csv` (when face teammate data is ready)
	- `data/processed/audio_features.csv` (when voice teammate data is ready)
4. Plots in `outputs/plots/` are present.
5. Branch `elvis` is pushed and up to date.

## Suggested Commit Message Format

Use one of these prefixes:

1. `feat:` new functionality
2. `fix:` bug fix
3. `docs:` documentation change
4. `chore:` maintenance/cleanup
5. `data:` dataset or generated artifact updates

Examples:

1. `feat(voice): add executed Task3 notebook with augmentation outputs`
2. `data(face): update image feature csv with member samples`
3. `docs: clarify branch workflow for teammate PR process`

## Quick FAQ

1. Do we push notebooks? Yes, push executed notebooks in `notebooks/`.
2. Can I edit another teammate's notebook? Only via PR and with explicit review.
3. What if optional package is missing? Install it, but notebooks also contain fallback behavior.
4. Which branch should final work target? `elvis` unless the team lead says otherwise.
