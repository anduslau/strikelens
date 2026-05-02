# StrikeLens

StrikeLens is a local Streamlit app for single-athlete combat-sports strike review and fine strike classification. The current product is locked around a Phase 19/22 workflow: detect strike candidates, review clips, correct labels, rebuild the fixed Phase 19 feature dataset, and train one fine strike classifier.

The current build does not support sparring, does not train a coarse model, and keeps prediction aligned to the saved fine model feature list.

## Current Workflow

1. Upload or select an MP4 video.
2. Extract MediaPipe pose landmarks into `data/processed`.
3. Detect generic strike-like events from pose movement.
4. Export short event clips for review.
5. Review clips and save corrected labels to `data/labels/event_labels.csv`.
6. Export the clean dataset from `Dataset & Label Review`.
7. The app automatically rebuilds `data/exports/ml_feature_dataset_v6.csv`.
8. Train the fine strike model from the `Fine Strike Model` page.
9. Use saved model suggestions during future clip review.

## Current Features

- MP4 upload and local video selection.
- Video preview and OpenCV metadata display.
- MediaPipe pose extraction to CSV.
- Optional annotated pose overlay video generation.
- Generic strike-candidate detection from wrist and ankle motion.
- Event clip export with FFmpeg H.264 conversion when available.
- Manual clip review with corrected category, corrected strike type, valid-strike flag, and notes.
- Model-assisted fine strike suggestions during review.
- Dataset QA, filtering, replay, and clean export.
- Automatic Phase 19 feature rebuild after clean export.
- Fine strike RandomForest training from exactly 80 selected features.
- Saved model metadata and freshness checks for clean labels, feature data, and trained model.

## Locked Model Contract

The fine strike model trains from:

- Dataset: `data/exports/ml_feature_dataset_v6.csv`
- Selected features: `data/exports/selected_features_corrected_strike_type_top_80_by_importance.csv`
- Target: `corrected_strike_type`
- Model: `ml/models/strike_type_random_forest.joblib`
- Metadata: `ml/models/strike_type_random_forest_metadata.json`
- Metrics: `data/exports/model_evaluation.csv`

Training and prediction use the exact saved feature list. During prediction, incoming features are aligned to the saved model columns: missing columns are filled with `0`, extra columns are dropped, and warnings are returned for mismatches.

## Strike Labels

Fine strike labels currently available in the review UI:

- `jab`
- `cross`
- `hook`
- `uppercut`
- `roundhouse_kick`
- `axe_kick`
- `back_kick`
- `cut_kick`
- `double_kick`
- `frontdouble_kick`
- `hopstep_kick`
- `spinninghook_kick`
- `tornado_kick`
- `hopaxe_kick`
- `cheapshot_kick`
- `crescentaxe_kick`
- `unknown`

The clean training export includes only rows where `is_valid_strike = true`, `corrected_category != unknown`, and `corrected_strike_type != unknown`.

## Project Structure

```text
strikelens/
|-- app.py
|-- README.md
|-- requirements.txt
|-- data/
|   |-- uploads/
|   |-- processed/
|   |-- processed/clips/
|   |-- labels/
|   |   |-- event_labels.csv
|   |-- exports/
|       |-- clean_labeled_events.csv
|       |-- ml_feature_dataset_v6.csv
|       |-- selected_features_corrected_strike_type_top_80_by_importance.csv
|       |-- model_evaluation.csv
|-- ml/
|   |-- build_feature_dataset.py
|   |-- evaluate_model.py
|   |-- feature_engineering.py
|   |-- predict_strike_type.py
|   |-- train_baseline_model.py
|   |-- models/
|       |-- strike_type_random_forest.joblib
|       |-- strike_type_random_forest_metadata.json
|-- src/
    |-- clip_exporter.py
    |-- pose_extractor.py
    |-- strike_detector.py
    |-- taxonomy.py
    |-- utils.py
    |-- video_loader.py
    |-- visualization.py
    |-- tracking/
```

## Key Files

- `app.py`: Streamlit workflow, review UI, clean export, automatic Phase 19 feature rebuild, and fine model training page.
- `ml/build_feature_dataset.py`: Builds event-window feature rows from clean labels plus processed pose/event CSVs.
- `ml/feature_engineering.py`: Computes the fixed event-level numeric features used by training and prediction.
- `ml/train_baseline_model.py`: Trains the locked fine strike RandomForest on the top 80 selected features.
- `ml/predict_strike_type.py`: Loads the saved model bundle and aligns incoming prediction features to its saved feature list.
- `src/clip_exporter.py`: Exports review clips and converts them with FFmpeg when available.
- `src/pose_extractor.py`: Extracts MediaPipe pose landmarks from video.
- `src/strike_detector.py`: Detects generic strike-like event windows from pose movement.
- `src/taxonomy.py`: Maintains the fine-to-coarse mapping used for exported data context.
- `src/utils.py`: Shared project paths.

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit.

## Notes

- The app is scoped to single-athlete videos.
- Coarse model training is intentionally hidden from the product workflow.
- The old `data/exports/ml_feature_dataset.csv` and coarse model artifacts may still exist as legacy files, but the active fine model workflow uses `ml_feature_dataset_v6.csv`.
- README content should be kept in sync with the locked workflow before future phase work.
