import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


cells = [
    md(
        """
# Quest 3 Student MoE: Eye + Head + Hand

This notebook trains the student model only. It does not use AU facial features, EmojiHero data, AU threshold logic, teacher TCN/Conv1D cells, or DANN cells.

Student inputs:
- eye features
- head features
- hand/controller features

Labels:
- `Yemo`: emotion class from the emotion folder name
- `Ydom`: recording domain, where `VideoRecordings = 0` and `ActingRecordings = 1`
"""
    ),
    code(
        """
from pathlib import Path
import os
import random

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
"""
    ),
    md("## Paths"),
    code(
        r"""
# Use the Colab path when mounted; otherwise use the local workspace data.
COLAB_DATA_ROOT = Path("/content/drive/MyDrive/Data Collection (New)")
LOCAL_DATA_ROOT = Path(r"C:\Users\alime\multimodal-vr-emotion-study\multimodal-vr-emotion-study-main")

if COLAB_DATA_ROOT.exists():
    data_folder_path = COLAB_DATA_ROOT
else:
    data_folder_path = LOCAL_DATA_ROOT

output_path = Path("/content/drive/MyDrive/outputs_10participants")
if not str(data_folder_path).startswith("/content"):
    output_path = data_folder_path / "student_outputs_10participants"

models_path = output_path / "models"
models_path.mkdir(parents=True, exist_ok=True)

print("Data folder:", data_folder_path)
print("Output folder:", output_path)
"""
    ),
    md("## Labels and Feature Columns"),
    code(
        """
EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"]
le = LabelEncoder().fit(EMOTIONS)

print("Emotion classes:", list(le.classes_))
print("Emotion mapping:", {name: int(le.transform([name])[0]) for name in le.classes_})

DOMAIN_LABELS = {
    "VideoRecordings": 0,   # natural
    "ActingRecordings": 1,  # acted
}

SEQ_LENGTH = 5
DOWNSAMPLE_FACTOR = 6  # 60 FPS -> 10 Hz
"""
    ),
    code(
        """
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

print("Eye features:", len(EYE_COLS))
print("Head features:", len(HEAD_COLS))
print("Hand features:", len(HAND_COLS))
print("All student features:", len(STUDENT_COLS))
"""
    ),
    md("## Build Recording Manifest"),
    code(
        """
def parse_recording_file(csv_path: Path):
    csv_path = Path(csv_path)
    recording_type = csv_path.parents[2].name
    participant_id = csv_path.parents[1].name
    emotion_name = csv_path.parent.name

    if recording_type not in DOMAIN_LABELS:
        raise ValueError(f"Unexpected recording type: {recording_type} in {csv_path}")
    if emotion_name not in le.classes_:
        raise ValueError(f"Unexpected emotion folder: {emotion_name} in {csv_path}")

    return {
        "path": csv_path,
        "participant": participant_id,
        "emotion": emotion_name,
        "Yemo": int(le.transform([emotion_name])[0]),
        "recording_type": recording_type,
        "Ydom": DOMAIN_LABELS[recording_type],
    }


acting_files = sorted((data_folder_path / "ActingRecordings").glob("*/*/acting.csv"))
video_files = sorted((data_folder_path / "VideoRecordings").glob("*/*/weights.csv"))
csv_files = acting_files + video_files

records = pd.DataFrame([parse_recording_file(path) for path in csv_files])

def missing_student_columns(csv_path: Path):
    columns = pd.read_csv(csv_path, nrows=0).columns
    return [col for col in STUDENT_COLS if col not in columns]


records["missing_student_cols"] = records["path"].apply(missing_student_columns)
excluded_records = records[records["missing_student_cols"].map(len).gt(0)].copy()
records = records[records["missing_student_cols"].map(len).eq(0)].reset_index(drop=True)

print("Acting files:", len(acting_files))
print("Video files:", len(video_files))
print("Total files before feature validation:", len(csv_files))
print("Usable files after feature validation:", len(records))
print("Excluded incomplete files:", len(excluded_records))
print("Participants:", records["participant"].nunique())
if len(excluded_records):
    display(excluded_records[["path", "participant", "emotion", "recording_type", "missing_student_cols"]])
display(records.head())
display(pd.crosstab(records["participant"], records["emotion"]))
display(pd.crosstab(records["recording_type"], records["emotion"]))
"""
    ),
    md("## Participant-Exclusive Split"),
    code(
        """
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)

records = records.copy()
records["fold"] = -1

for fold_id, (_, heldout_idx) in enumerate(
    sgkf.split(records["path"], records["Yemo"], groups=records["participant"])
):
    records.loc[heldout_idx, "fold"] = fold_id

train_records = records[records["fold"].between(0, 6)].reset_index(drop=True)
val_records = records[records["fold"].eq(7)].reset_index(drop=True)
test_records = records[records["fold"].between(8, 9)].reset_index(drop=True)

print("Train participants:", sorted(train_records["participant"].unique()))
print("Val participants:", sorted(val_records["participant"].unique()))
print("Test participants:", sorted(test_records["participant"].unique()))
print()
print("Recording counts:", {
    "train": len(train_records),
    "val": len(val_records),
    "test": len(test_records),
})

assert set(train_records["participant"]).isdisjoint(val_records["participant"])
assert set(train_records["participant"]).isdisjoint(test_records["participant"])
assert set(val_records["participant"]).isdisjoint(test_records["participant"])
assert train_records["participant"].nunique() == 7
assert val_records["participant"].nunique() == 1
assert test_records["participant"].nunique() == 2
"""
    ),
    md("## Downsample and Window"),
    code(
        """
def downsample_60hz_to_10hz(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0)

    usable_rows = (len(features) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    features = features.iloc[:usable_rows]

    downsampled = (
        features
        .groupby(np.arange(len(features)) // DOWNSAMPLE_FACTOR)
        .mean()
        .reset_index(drop=True)
    )
    return downsampled


def make_student_windows(df_10hz: pd.DataFrame, yemo: int, ydom: int, seq_length: int = SEQ_LENGTH):
    X_eye, X_head, X_hand, Yemo, Ydom = [], [], [], [], []

    usable_rows = (len(df_10hz) // seq_length) * seq_length
    df_10hz = df_10hz.iloc[:usable_rows]

    for start in range(0, usable_rows, seq_length):
        window = df_10hz.iloc[start:start + seq_length]
        X_eye.append(window[EYE_COLS].to_numpy(dtype=np.float32))
        X_head.append(window[HEAD_COLS].to_numpy(dtype=np.float32))
        X_hand.append(window[HAND_COLS].to_numpy(dtype=np.float32))
        Yemo.append(yemo)
        Ydom.append(ydom)

    return X_eye, X_head, X_hand, Yemo, Ydom


def load_recording_windows(row):
    df = pd.read_csv(row["path"])
    df_10hz = downsample_60hz_to_10hz(df, STUDENT_COLS)
    return make_student_windows(df_10hz, row["Yemo"], row["Ydom"])


def load_split(record_df: pd.DataFrame):
    X_eye, X_head, X_hand, Yemo, Ydom = [], [], [], [], []

    for _, row in record_df.iterrows():
        eye, head, hand, yemo, ydom = load_recording_windows(row)
        X_eye.extend(eye)
        X_head.extend(head)
        X_hand.extend(hand)
        Yemo.extend(yemo)
        Ydom.extend(ydom)

    return (
        np.asarray(X_eye, dtype=np.float32),
        np.asarray(X_head, dtype=np.float32),
        np.asarray(X_hand, dtype=np.float32),
        np.asarray(Yemo, dtype=np.int64),
        np.asarray(Ydom, dtype=np.int64),
    )
"""
    ),
    code(
        """
X_eye_train, X_head_train, X_hand_train, Yemo_train, Ydom_train = load_split(train_records)
X_eye_val, X_head_val, X_hand_val, Yemo_val, Ydom_val = load_split(val_records)
X_eye_test, X_head_test, X_hand_test, Yemo_test, Ydom_test = load_split(test_records)

print("X_eye_train:", X_eye_train.shape)
print("X_head_train:", X_head_train.shape)
print("X_hand_train:", X_hand_train.shape)
print("Yemo_train:", Yemo_train.shape)
print("Ydom_train:", Ydom_train.shape)
print()
print("X_eye_val:", X_eye_val.shape)
print("X_head_val:", X_head_val.shape)
print("X_hand_val:", X_hand_val.shape)
print()
print("X_eye_test:", X_eye_test.shape)
print("X_head_test:", X_head_test.shape)
print("X_hand_test:", X_hand_test.shape)

assert X_eye_train.shape[1:] == (SEQ_LENGTH, len(EYE_COLS))
assert X_head_train.shape[1:] == (SEQ_LENGTH, len(HEAD_COLS))
assert X_hand_train.shape[1:] == (SEQ_LENGTH, len(HAND_COLS))
"""
    ),
    md("## Student MoE Model"),
    code(
        """
def build_temporal_expert(input_tensor, name: str, units: int = 64, dropout: float = 0.25):
    x = layers.Masking(mask_value=0.0, name=f"{name}_mask")(input_tensor)
    x = layers.Conv1D(units, kernel_size=3, padding="same", activation="relu", name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv1D(units, kernel_size=3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    x = layers.Dense(units, activation="relu", name=f"{name}_dense")(x)
    return x


def build_student_moe(seq_length=SEQ_LENGTH, num_classes=len(le.classes_)):
    eye_input = layers.Input(shape=(seq_length, len(EYE_COLS)), name="eye_input")
    head_input = layers.Input(shape=(seq_length, len(HEAD_COLS)), name="head_input")
    hand_input = layers.Input(shape=(seq_length, len(HAND_COLS)), name="hand_input")

    eye_expert = build_temporal_expert(eye_input, "eye_expert")
    head_expert = build_temporal_expert(head_input, "head_expert")
    hand_expert = build_temporal_expert(hand_input, "hand_expert")

    merged_sequence = layers.Concatenate(axis=-1, name="student_signal_sequence")(
        [eye_input, head_input, hand_input]
    )
    shared_expert = build_temporal_expert(merged_sequence, "shared_expert")

    experts = layers.Concatenate(axis=1, name="expert_stack")([
        layers.Reshape((1, 64), name="eye_expert_axis")(eye_expert),
        layers.Reshape((1, 64), name="head_expert_axis")(head_expert),
        layers.Reshape((1, 64), name="hand_expert_axis")(hand_expert),
        layers.Reshape((1, 64), name="shared_expert_axis")(shared_expert),
    ])

    gate_features = layers.Concatenate(name="gate_features")(
        [eye_expert, head_expert, hand_expert, shared_expert]
    )
    gate = layers.Dense(64, activation="relu", name="gate_hidden")(gate_features)
    gate = layers.Dense(4, activation="softmax", name="expert_gate")(gate)

    moe = layers.Dot(axes=(1, 1), name="moe_weighted_sum")([gate, experts])
    moe = layers.Dropout(0.3, name="moe_dropout")(moe)
    moe = layers.Dense(128, activation="relu", name="classifier_hidden")(moe)
    emotion_output = layers.Dense(num_classes, activation="softmax", name="emotion_output")(moe)

    model = models.Model(
        inputs=[eye_input, head_input, hand_input],
        outputs=emotion_output,
        name="student_MoE_eye_head_hand",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


student_model = build_student_moe()
student_model.summary()
"""
    ),
    md("## Train"),
    code(
        """
early_stop = callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=8,
    restore_best_weights=True,
)

checkpoint_path = models_path / "student_MoE_eye_head_hand_10participants_10hz.best.keras"
checkpoint = callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_accuracy",
    save_best_only=True,
)

history = student_model.fit(
    [X_eye_train, X_head_train, X_hand_train],
    Yemo_train,
    validation_data=([X_eye_val, X_head_val, X_hand_val], Yemo_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop, checkpoint],
)
"""
    ),
    md("## Evaluate Participant-Held-Out Test Set"),
    code(
        """
test_loss, test_accuracy = student_model.evaluate(
    [X_eye_test, X_head_test, X_hand_test],
    Yemo_test,
    verbose=0,
)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

Yemo_prob = student_model.predict([X_eye_test, X_head_test, X_hand_test])
Yemo_pred = np.argmax(Yemo_prob, axis=1)

print(classification_report(
    Yemo_test,
    Yemo_pred,
    labels=np.arange(len(le.classes_)),
    target_names=le.classes_,
    zero_division=0,
))

cm = confusion_matrix(Yemo_test, Yemo_pred, labels=np.arange(len(le.classes_)))
display(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
"""
    ),
    md("## Save Final Model"),
    code(
        """
final_model_path = models_path / "student_MoE_eye_head_hand_10participants_10hz.keras"
student_model.save(final_model_path)
print("Saved model:", final_model_path)
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    output = Path(
        r"C:\Users\alime\multimodal-vr-emotion-study\student_MoE_eye_head_hand_10participants_10hz.ipynb"
    )
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
