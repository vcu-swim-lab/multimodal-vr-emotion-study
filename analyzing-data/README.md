# Analyzing Data — VR Emotion Study

Interactive Streamlit dashboard for exploring VR emotion study recordings.

## Requirements

- Python 3.10+
- Data folders from the study:
  ```
  multimodal-vr-emotion-study-main/
  ├── VideoRecordings/{session_id}/{emotion}/weights.csv
  └── ActingRecordings/{session_id}/{emotion}/acting.csv
  ```

## Setup

```bash
cd analyzing-data
python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

## Dashboard tabs

| Tab | Content |
|-----|---------|
| **Face** | Facial Action Unit weights over time + mean per emotion |
| **Head** | Head X/Y/Z position over time + speed distribution |
| **Hands** | Left/Right hand position over time + speed distribution |
| **Controllers** | Left/Right controller speed distribution |
| **Quality** | Tracking rate (%) per signal, emotion, and phase |

## Sidebar controls

- **Project root path** — path to `multimodal-vr-emotion-study-main/`. Defaults to `../multimodal-vr-emotion-study-main` relative to this folder.
- **Session** — auto-discovered from `VideoRecordings/` subdirectories.
- **Phase** — filter by Video, Acting, or both.
- **Emotions** — select/deselect individual emotions.

## Notebook

`notebooks/00_explore.ipynb` — original exploratory analysis with static matplotlib/seaborn plots and CSV export.
