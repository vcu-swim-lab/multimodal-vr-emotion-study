import json
from pathlib import Path


SOURCE = Path(r"C:\Users\alime\Downloads\student_MoE_eye_head_hand_10participants (2).ipynb")
OUTPUT = Path(
    r"C:\Users\alime\multimodal-vr-emotion-study\student_MoE_eye_head_hand_10participants_finished.ipynb"
)


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"}


def code(text: str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip() + "\n",
    }


nb = json.loads(SOURCE.read_text(encoding="utf-8"))

# Keep the already-converted student setup cells, but remove the TCN install cell.
cells = []
cells.append(md("# Student MoE: Quest 3 Eye + Head + Hand"))
cells.append(nb["cells"][1])
cells.append(nb["cells"][2])
cells.append(nb["cells"][4])
cells.extend(nb["cells"][5:19])

# Make the split cell more verbose so the user can verify participant exclusivity.
cells[15] = code(
    """
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)

records["fold"] = -1

for fold_id, (_, heldout_idx) in enumerate(
    sgkf.split(records["path"], records["Yemo"], groups=records["participant"])
):
    records.loc[heldout_idx, "fold"] = fold_id

train_records = records[records["fold"].between(0, 6)].reset_index(drop=True)
val_records = records[records["fold"] == 7].reset_index(drop=True)
test_records = records[records["fold"].between(8, 9)].reset_index(drop=True)

print("Train participants:", sorted(train_records["participant"].unique()))
print("Val participants:", sorted(val_records["participant"].unique()))
print("Test participants:", sorted(test_records["participant"].unique()))
print()
print("Train participant count:", train_records["participant"].nunique())
print("Val participant count:", val_records["participant"].nunique())
print("Test participant count:", test_records["participant"].nunique())

assert train_records["participant"].nunique() == 7
assert val_records["participant"].nunique() == 1
assert test_records["participant"].nunique() == 2
assert set(train_records["participant"]).isdisjoint(val_records["participant"])
assert set(train_records["participant"]).isdisjoint(test_records["participant"])
assert set(val_records["participant"]).isdisjoint(test_records["participant"])

print("Train distribution:")
display(pd.crosstab([train_records["participant"], train_records["recording_type"]], train_records["emotion"]))

print("Validation distribution:")
display(pd.crosstab([val_records["participant"], val_records["recording_type"]], val_records["emotion"]))

print("Test distribution:")
display(pd.crosstab([test_records["participant"], test_records["recording_type"]], test_records["emotion"]))
"""
)

cells.extend(
    [
        md("#9. Make Non-Overlapping Windows"),
        code(
            """
def make_student_windows(df_10hz, yemo, ydom, seq_length=SEQ_LENGTH):
    X_eye, X_head, X_hand = [], [], []
    Yemo, Ydom = [], []

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
"""
        ),
        md("#10. Load Split Data"),
        code(
            """
def load_recording_windows(row):
    df = pd.read_csv(row["path"])
    df_10hz = downsample_60hz_to_10hz(df, STUDENT_COLS)
    return make_student_windows(df_10hz, row["Yemo"], row["Ydom"])


def load_split(record_df):
    X_eye, X_head, X_hand = [], [], []
    Yemo, Ydom = [], []

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
        md("#11. Create Train / Validation / Test Arrays"),
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
print("Yemo_val:", Yemo_val.shape)
print()
print("X_eye_test:", X_eye_test.shape)
print("X_head_test:", X_head_test.shape)
print("X_hand_test:", X_hand_test.shape)
print("Yemo_test:", Yemo_test.shape)

assert X_eye_train.shape[1:] == (SEQ_LENGTH, len(EYE_COLS))
assert X_head_train.shape[1:] == (SEQ_LENGTH, len(HEAD_COLS))
assert X_hand_train.shape[1:] == (SEQ_LENGTH, len(HAND_COLS))
"""
        ),
        md("#12. Student MoE Model"),
        code(
            """
def build_expert(input_tensor, name, units=64, dropout=0.25):
    x = layers.Conv1D(units, 3, padding="same", activation="relu", name=f"{name}_conv1")(input_tensor)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Conv1D(units, 3, padding="same", activation="relu", name=f"{name}_conv2")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    x = layers.Dense(units, activation="relu", name=f"{name}_dense")(x)
    return x


def build_student_moe():
    eye_input = layers.Input(shape=(SEQ_LENGTH, len(EYE_COLS)), name="eye_input")
    head_input = layers.Input(shape=(SEQ_LENGTH, len(HEAD_COLS)), name="head_input")
    hand_input = layers.Input(shape=(SEQ_LENGTH, len(HAND_COLS)), name="hand_input")

    eye_expert = build_expert(eye_input, "eye_expert")
    head_expert = build_expert(head_input, "head_expert")
    hand_expert = build_expert(hand_input, "hand_expert")

    merged = layers.Concatenate(axis=-1, name="student_signal_sequence")(
        [eye_input, head_input, hand_input]
    )
    shared_expert = build_expert(merged, "shared_expert")

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
    x = layers.Dropout(0.3, name="moe_dropout")(moe)
    x = layers.Dense(128, activation="relu", name="classifier_hidden")(x)
    emotion_output = layers.Dense(len(le.classes_), activation="softmax", name="emotion_output")(x)

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
        md("#13. Train"),
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
        md("#14. Evaluate"),
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
        md("#15. Save Final Model"),
        code(
            """
final_model_path = models_path / "student_MoE_eye_head_hand_10participants_10hz.keras"
student_model.save(final_model_path)
print("Saved model:", final_model_path)
"""
        ),
    ]
)

nb["cells"] = cells
OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUTPUT)
