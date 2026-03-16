# Teammate Quickstart 101

This is the single official onboarding document for teammates.

If you follow this file exactly, you will be able to:

1. Set up your environment.
2. Work safely in your own branch.
3. Run only your task notebook correctly.
4. Commit only the right files.
5. Push to GitHub without breaking others.
6. Open a clean PR for integration into the elvis branch.

## 1. Project Summary

This project has four assignment tasks:

1. Task 1: Product recommendation model (Elvis primary ownership).
2. Task 2: Face image processing workflow.
3. Task 3: Voice audio processing workflow.
4. Task 4: Multimodal integration demo (face gate -> voice gate -> product recommendation).

Main notebook files:

1. notebooks/Task1_Product_Recommendation.ipynb
2. notebooks/Task2_Face_Image_Processing.ipynb
3. notebooks/Task3_Voice_Audio_Processing.ipynb
4. notebooks/Task4_Multimodal_Integration.ipynb

## 2. Team Branch Rules (Very Important)

1. Do not work directly on main.
2. Do not work directly on elvis unless you are integrating approved PRs.
3. Each teammate works in a personal branch (example: alice-face, brian-voice).
4. Always pull latest elvis before starting work.
5. Open PR from your personal branch to elvis.

Recommended branch model:

1. main = stable baseline.
2. elvis = assignment integration branch.
3. teammate branches = personal development branches.

## 3. One-Time Setup on Your Machine

## Step 1: Clone repository

```bash
git clone https://github.com/Elvis-Kayonga/Data-Preprocessing-Formative2-ML-GROUP9.git
cd Data-Preprocessing-Formative2-ML-GROUP9
```

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Confirm remotes and branches

```bash
git remote -v
git fetch --all
git branch -a
```

Expected remote includes origin pointing to GitHub repo URL.

## 4. Daily Start Workflow (Run Every Time Before You Code)

```bash
git fetch origin
git checkout elvis
git pull origin elvis
```

Create or refresh your personal branch from latest elvis:

```bash
git checkout -B <your-branch-name>
```

Examples:

1. git checkout -B ruth-face
2. git checkout -B john-voice
3. git checkout -B mercy-integration

## 5. Role-Based Task Guide

## A. Face teammate (Task 2)

### Input you must provide

Add image files here:

1. data/images/<your_name>/

Use at least 3 clear face images:

1. neutral expression
2. smiling expression
3. surprised expression

Supported formats:

1. .jpg
2. .png

### Run notebook

1. Open notebooks/Task2_Face_Image_Processing.ipynb
2. Run all cells top to bottom.

### Expected outputs

1. data/processed/image_features.csv
2. outputs/plots/10_face_preview.png (if images found)
3. outputs/plots/11_face_augmentation.png (if images found)

### If OpenCV error appears

```bash
pip install opencv-python scikit-image
```

## B. Voice teammate (Task 3)

### Input you must provide

Add audio files here:

1. data/audio/<your_name>/

Use at least 3 clips:

1. 2 to 5 seconds each
2. .wav format
3. clean speech, low noise

### Run notebook

1. Open notebooks/Task3_Voice_Audio_Processing.ipynb
2. Run all cells top to bottom.

### Expected outputs

1. data/processed/audio_features.csv
2. outputs/plots/12_audio_waveform.png (if audio found)
3. outputs/plots/13_audio_augmentation.png (if audio found)

### If librosa error appears

```bash
pip install librosa soundfile
```

## C. Integration teammate (Task 4)

### Prerequisites

1. Task 1 model exists in models/product_recommendation_model.joblib
2. Task 2 and Task 3 outputs are ideally available.

### Run notebook

1. Open notebooks/Task4_Multimodal_Integration.ipynb
2. Run all cells top to bottom.

### Expected outputs

1. gate decision scenarios printed
2. outputs/plots/14_multimodal_gate_results.png

## 6. Safe Commit Workflow (Do Not Skip)

## Step 1: Check changed files

```bash
git status
```

## Step 2: Stage only files for your task

Example for face teammate:

```bash
git add notebooks/Task2_Face_Image_Processing.ipynb
git add data/processed/image_features.csv
git add outputs/plots/10_face_preview.png outputs/plots/11_face_augmentation.png
```

Example for voice teammate:

```bash
git add notebooks/Task3_Voice_Audio_Processing.ipynb
git add data/processed/audio_features.csv
git add outputs/plots/12_audio_waveform.png outputs/plots/13_audio_augmentation.png
```

## Step 3: Commit with clear message

```bash
git commit -m "feat(face): add executed Task2 notebook and image features"
```

More message examples:

1. feat(voice): add executed Task3 notebook and audio features
2. fix(face): correct image path and rerun notebook
3. docs(team): clarify teammate setup steps
4. data(voice): refresh audio feature csv with new samples

## 7. Push Workflow

Push your branch first time:

```bash
git push -u origin <your-branch-name>
```

Push subsequent updates:

```bash
git push origin <your-branch-name>
```

## 8. Pull Request Workflow

After pushing branch:

1. Go to GitHub repo.
2. Create PR from <your-branch-name> to elvis.
3. Add PR title and description.
4. Request reviewer.

PR description template:

1. Summary of changes.
2. Notebook executed.
3. Output artifacts generated.
4. Any known limitation.

Example:

1. Updated Task3 voice notebook with real wav samples.
2. Executed full notebook and exported audio_features.csv.
3. Generated waveform and augmentation plots.
4. No runtime errors.

## 9. If Push Is Rejected

If non-fast-forward happens:

```bash
git fetch origin
git pull --rebase origin <your-branch-name>
git push origin <your-branch-name>
```

If rebase conflicts appear:

1. open conflicted files.
2. resolve markers.
3. git add <file>
4. git rebase --continue
5. repeat until rebase ends.
6. push again.

## 10. If Your Branch Is Behind Elvis

To update your branch with latest elvis:

```bash
git fetch origin
git checkout <your-branch-name>
git rebase origin/elvis
```

If rebase is not preferred by your team, use merge:

```bash
git fetch origin
git checkout <your-branch-name>
git merge origin/elvis
```

## 11. Files You Must Not Randomly Edit

Only edit what belongs to your role unless coordinated:

1. notebooks/Task1_Product_Recommendation.ipynb (owner: Elvis)
2. src/data_pipeline.py and src/train_product_model.py (core pipeline)
3. requirements.txt (only if absolutely necessary)

## 12. Mandatory Verification Before PR

Run this checklist:

1. Notebook runs from first to last cell.
2. No red error output in notebook.
3. New files are relevant to your task.
4. No temporary scripts included.
5. git status is clean after commit.

Helpful checks:

```bash
git status
git log --oneline -n 5
```

## 13. Final Integration Flow (For Branch Owner)

For maintainer integrating teammate PRs into elvis:

1. Merge one PR at a time.
2. After each merge run:

```bash
python src/run_all.py
python -m nbconvert --to notebook --execute notebooks/Task4_Multimodal_Integration.ipynb --output Task4_Multimodal_Integration.ipynb --output-dir notebooks
```

3. If checks pass, push updated elvis.

## 14. Common Mistakes and Fixes

1. Mistake: Committed on main.
Fix: create new branch from main state, cherry-pick commit, reset main back.

2. Mistake: Added unrelated files (temp scripts, caches).
Fix: git restore --staged <file> then remove file locally.

3. Mistake: Notebook not executed before commit.
Fix: rerun notebook fully, save, recommit.

4. Mistake: Push rejected.
Fix: fetch + rebase + push as shown above.

5. Mistake: Wrong target branch in PR.
Fix: change PR base branch to elvis.

## 15. Submission-Ready Evidence Checklist

At submission time, confirm:

1. all four task notebooks exist and are executed.
2. processed csv outputs exist.
3. model artifacts exist.
4. required plots exist.
5. branch elvis is up to date on GitHub.

Core files to verify:

1. models/product_recommendation_model.joblib
2. models/product_model_metrics.json
3. data/processed/merged_customer_dataset.csv
4. data/processed/image_features.csv
5. data/processed/audio_features.csv
6. notebooks/Task1_Product_Recommendation.ipynb
7. notebooks/Task2_Face_Image_Processing.ipynb
8. notebooks/Task3_Voice_Audio_Processing.ipynb
9. notebooks/Task4_Multimodal_Integration.ipynb

## 16. Quick Command Reference

Start work:

```bash
git fetch origin
git checkout elvis
git pull origin elvis
git checkout -B <your-branch-name>
```

Commit work:

```bash
git status
git add <files>
git commit -m "type(scope): short message"
```

Push work:

```bash
git push -u origin <your-branch-name>
```

Sync if behind:

```bash
git fetch origin
git pull --rebase origin <your-branch-name>
```

## 17. Final Note

Work small, commit often, and push frequently.

If everyone follows this 101 guide, branch conflicts reduce a lot and final submission becomes smooth.
