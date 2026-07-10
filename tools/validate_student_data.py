from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path(r"C:\Users\alime\multimodal-vr-emotion-study\multimodal-vr-emotion-study-main")
SEED = 42
SEQ_LENGTH = 5
DOWNSAMPLE_FACTOR = 6

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
EMOTION_TO_ID = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}
DOMAIN_LABELS = {"VideoRecordings": 0, "ActingRecordings": 1}

EYE_COLS = [
    "EyeGazeOriginX",
    "EyeGazeOriginY",
    "EyeGazeOriginZ",
    "EyeGazeDirX",
    "EyeGazeDirY",
    "EyeGazeDirZ",
    "EyeGazePitch",
    "EyeGazeYaw",
    "EyeGazeRoll",
    "EyeGazeRotX",
    "EyeGazeRotY",
    "EyeGazeRotZ",
    "EyeGazeRotW",
]

HEAD_COLS = [
    "HeadX",
    "HeadY",
    "HeadZ",
    "HeadPitch",
    "HeadYaw",
    "HeadRoll",
    "HeadRotX",
    "HeadRotY",
    "HeadRotZ",
    "HeadRotW",
]

HAND_COLS = [
    "LeftHandX",
    "LeftHandY",
    "LeftHandZ",
    "RightHandTracked",
    "RightHandX",
    "RightHandY",
    "RightHandZ",
]

STUDENT_COLS = EYE_COLS + HEAD_COLS + HAND_COLS


def parse_recording_file(csv_path: Path):
    csv_path = Path(csv_path)
    recording_type = csv_path.parents[2].name
    participant_id = csv_path.parents[1].name
    emotion_name = csv_path.parent.name
    return {
        "path": csv_path,
        "participant": participant_id,
        "emotion": emotion_name,
        "Yemo": EMOTION_TO_ID[emotion_name],
        "recording_type": recording_type,
        "Ydom": DOMAIN_LABELS[recording_type],
    }


def downsample_60hz_to_10hz(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    usable_rows = (len(features) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    features = features.iloc[:usable_rows]
    return (
        features.groupby(np.arange(len(features)) // DOWNSAMPLE_FACTOR)
        .mean()
        .reset_index(drop=True)
    )


def count_windows(row) -> int:
    df = pd.read_csv(row["path"])
    df_10hz = downsample_60hz_to_10hz(df, STUDENT_COLS)
    return len(df_10hz) // SEQ_LENGTH


acting_files = sorted((DATA_ROOT / "ActingRecordings").glob("*/*/acting.csv"))
video_files = sorted((DATA_ROOT / "VideoRecordings").glob("*/*/weights.csv"))
records = pd.DataFrame([parse_recording_file(path) for path in acting_files + video_files])


def missing_student_columns(csv_path: Path):
    columns = pd.read_csv(csv_path, nrows=0).columns
    return [col for col in STUDENT_COLS if col not in columns]


records["missing_student_cols"] = records["path"].apply(missing_student_columns)
excluded_records = records[records["missing_student_cols"].map(len).gt(0)].copy()
records = records[records["missing_student_cols"].map(len).eq(0)].reset_index(drop=True)

print("usable files", len(records))
print("excluded incomplete files", len(excluded_records))
if len(excluded_records):
    print(excluded_records[["path", "participant", "emotion", "recording_type", "missing_student_cols"]])
print("participants", records["participant"].nunique())
print("emotion mapping", EMOTION_TO_ID)

sample = pd.read_csv(records.iloc[0]["path"])
missing = [col for col in STUDENT_COLS if col not in sample.columns]
print("sample missing columns", missing)

all_windows = int(sum(count_windows(row) for _, row in records.iterrows()))
print("all recordings", len(records), "windows", all_windows)
print("recordings by domain")
print(pd.crosstab(records["recording_type"], records["emotion"]))
print("recordings by participant")
print(records["participant"].value_counts().sort_index())

print("expected shapes")
print("X_eye:  (samples, 5, 13)")
print("X_head: (samples, 5, 10)")
print("X_hand: (samples, 5, 7)")
