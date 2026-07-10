import json
from pathlib import Path


SOURCE = Path(r"C:\Users\alime\Downloads\rq2_distillation (2).ipynb")
OUTPUT = Path(r"C:\Users\alime\multimodal-vr-emotion-study\rq2_distillation_ablation_eye_head_hand.ipynb")


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

# Keep the data imports, path setup, participant split, student feature definitions,
# optional threshold windowing, and data-shape review from the current RQ2 notebook.
# Stop before its partial distillation section; the complete RQ2 ablation follows below.
cells = nb["cells"][:27]

for cell in cells:
    if cell.get("cell_type") != "code":
        continue

    source = "".join(cell.get("source", []))
    if "from pathlib import Path" in source and "from functools import reduce" not in source:
        source = source.replace("from pathlib import Path\n", "from pathlib import Path\nfrom functools import reduce\n")

    source = source.replace(
        "USE_STUDENT_THRESHOLD = False  # set False to disable filtering entirely",
        "USE_STUDENT_THRESHOLD = True  # RQ2 uses reviewed student-signal threshold windowing",
    )
    source = source.replace(
        'print("Review passed: student uses eye/head/hand only, with teacher-style scan logic and threshold=0.25.")',
        'print(f"Review passed: student uses eye/head/hand only; thresholding enabled={USE_STUDENT_THRESHOLD}, threshold={STUDENT_THRESHOLD}.")',
    )
    cell["source"] = source

cells.extend(
    [
        md(
            """
# RQ2 Distillation Ablation

This section trains three student distillation schemes on the same participant-exclusive split:

- Flat fusion: concatenate eye, head, and hand at the input, no routing, soft-label loss only.
- Coarse MoE: domain-level gating with natural, acted, and shared experts, soft-label plus hard-label loss.
- EMOE with Unimodal Distillation: sample-conditional modality router plus standalone expert calibration losses.

Teacher soft labels are loaded from precomputed `.npy` files when available, otherwise they are generated from a saved teacher model using aligned face-AU windows from the same recording starts as the student windows.
"""
        ),
        code(
            r"""
# Teacher-compatible no-eye face AU columns from the teacher notebook.
AU_NAMES = [
    "BrowLowererL", "BrowLowererR", "CheekPuffL", "CheekPuffR",
    "CheekRaiserL", "CheekRaiserR", "CheekSuckL", "CheekSuckR",
    "ChinRaiserB", "ChinRaiserT", "DimplerL", "DimplerR",
    "InnerBrowRaiserL", "InnerBrowRaiserR", "JawDrop", "JawSidewaysLeft",
    "JawSidewaysRight", "JawThrust", "LidTightenerL", "LidTightenerR",
    "LipCornerDepressorL", "LipCornerDepressorR", "LipCornerPullerL", "LipCornerPullerR",
    "LipFunnelerLB", "LipFunnelerLT", "LipFunnelerRB", "LipFunnelerRT",
    "LipPressorL", "LipPressorR", "LipPuckerL", "LipPuckerR",
    "LipStretcherL", "LipStretcherR", "LipSuckLB", "LipSuckLT",
    "LipSuckRB", "LipSuckRT", "LipTightenerL", "LipTightenerR",
    "LipsToward", "LowerLipDepressorL", "LowerLipDepressorR", "MouthLeft",
    "MouthRight", "NoseWrinklerL", "NoseWrinklerR", "OuterBrowRaiserL",
    "OuterBrowRaiserR", "UpperLidRaiserL", "UpperLidRaiserR", "UpperLipRaiserL",
    "UpperLipRaiserR", "TongueTipInterdental", "TongueTipAlveolar",
    "TongueFrontDorsalPalate", "TongueMidDorsalPalate", "TongueBackDorsalVelar",
    "TongueOut", "TongueRetreat",
]

assert len(AU_NAMES) == 60
assert not any(col.startswith("Eyes") for col in AU_NAMES)

TEACHER_MODEL_PATH = models_path / "teacher-no-eye-AU-acted-natural-MoE_3experts_2gate_with_10participants_data.keras"
TEACHER_PROB_DIR = output_path / "teacher_soft_labels_rq2"
TEACHER_TEMPERATURE = 2.0

DISTILLATION_WEIGHT_GRID = [
    {"soft": 1.0, "hard": 0.25, "unimodal": 0.10},
    {"soft": 1.0, "hard": 0.50, "unimodal": 0.25},
    {"soft": 1.0, "hard": 1.00, "unimodal": 0.50},
]

print("Teacher model path:", TEACHER_MODEL_PATH)
print("Teacher probability dir:", TEACHER_PROB_DIR)
print("Teacher AU features:", len(AU_NAMES))
"""
        ),
        md("## Rebuild Aligned Student and Teacher Windows"),
        code(
            """
def downsample_60hz_to_10hz_numeric(df, feature_cols):
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing[:10]} total={len(missing)}")

    features = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0)

    usable_rows = (len(features) // DOWNSAMPLE_FACTOR) * DOWNSAMPLE_FACTOR
    features = features.iloc[:usable_rows]

    return (
        features
        .groupby(np.arange(len(features)) // DOWNSAMPLE_FACTOR)
        .mean()
        .reset_index(drop=True)
    )


def make_aligned_student_teacher_windows(row):
    raw = pd.read_csv(row["path"])
    student_df = downsample_60hz_to_10hz(raw, STUDENT_COLS)
    teacher_df = downsample_60hz_to_10hz_numeric(raw, AU_NAMES)

    usable = min(len(student_df), len(teacher_df))
    student_df = student_df.iloc[:usable].reset_index(drop=True)
    teacher_df = teacher_df.iloc[:usable].reset_index(drop=True)

    starts = student_style_keep_starts(student_df, row["emotion"], seq_length=SEQ_LENGTH)

    X_eye, X_head, X_hand, X_teacher, Yemo, Ydom = [], [], [], [], [], []
    for start in starts:
        student_window = student_df.iloc[start:start + SEQ_LENGTH]
        teacher_window = teacher_df.iloc[start:start + SEQ_LENGTH]

        if len(student_window) != SEQ_LENGTH or len(teacher_window) != SEQ_LENGTH:
            continue

        X_eye.append(student_window[EYE_COLS].to_numpy(dtype=np.float32))
        X_head.append(student_window[HEAD_COLS].to_numpy(dtype=np.float32))
        X_hand.append(student_window[HAND_COLS].to_numpy(dtype=np.float32))
        X_teacher.append(teacher_window[AU_NAMES].to_numpy(dtype=np.float32))
        Yemo.append(row["Yemo"])
        Ydom.append(row["Ydom"])

    return X_eye, X_head, X_hand, X_teacher, Yemo, Ydom


def load_aligned_split(record_df):
    X_eye, X_head, X_hand, X_teacher, Yemo, Ydom = [], [], [], [], [], []

    for _, row in record_df.iterrows():
        eye, head, hand, teacher, yemo, ydom = make_aligned_student_teacher_windows(row)
        X_eye.extend(eye)
        X_head.extend(head)
        X_hand.extend(hand)
        X_teacher.extend(teacher)
        Yemo.extend(yemo)
        Ydom.extend(ydom)

    return (
        np.asarray(X_eye, dtype=np.float32),
        np.asarray(X_head, dtype=np.float32),
        np.asarray(X_hand, dtype=np.float32),
        np.asarray(X_teacher, dtype=np.float32),
        np.asarray(Yemo, dtype=np.int64),
        np.asarray(Ydom, dtype=np.int64),
    )


X_eye_train, X_head_train, X_hand_train, X_teacher_train, Yemo_train, Ydom_train = load_aligned_split(train_records)
X_eye_val, X_head_val, X_hand_val, X_teacher_val, Yemo_val, Ydom_val = load_aligned_split(val_records)
X_eye_test, X_head_test, X_hand_test, X_teacher_test, Yemo_test, Ydom_test = load_aligned_split(test_records)

X_flat_train = np.concatenate([X_eye_train, X_head_train, X_hand_train], axis=-1)
X_flat_val = np.concatenate([X_eye_val, X_head_val, X_hand_val], axis=-1)
X_flat_test = np.concatenate([X_eye_test, X_head_test, X_hand_test], axis=-1)

print("Student train:", X_eye_train.shape, X_head_train.shape, X_hand_train.shape)
print("Teacher train:", X_teacher_train.shape)
print("Flat train:", X_flat_train.shape)
print("Labels:", Yemo_train.shape, Ydom_train.shape)

assert X_flat_train.shape[1:] == (SEQ_LENGTH, len(STUDENT_COLS))
assert X_teacher_train.shape[1:] == (SEQ_LENGTH, len(AU_NAMES))
assert len(X_flat_train) == len(Yemo_train) == len(Ydom_train)
"""
        ),
        md("## Teacher Soft Labels"),
        code(
            """
def soften_probabilities(probabilities, temperature=1.0):
    probabilities = np.asarray(probabilities, dtype=np.float32)
    probabilities = np.clip(probabilities, 1e-8, 1.0)

    if temperature == 1.0:
        softened = probabilities
    else:
        logits = np.log(probabilities) / float(temperature)
        logits = logits - logits.max(axis=1, keepdims=True)
        softened = np.exp(logits)
        softened = softened / softened.sum(axis=1, keepdims=True)

    return softened.astype(np.float32)


def load_teacher_model():
    if not Path(TEACHER_MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Teacher model not found: {TEACHER_MODEL_PATH}. "
            "Set TEACHER_MODEL_PATH to the saved teacher .keras file, or place "
            "Ysoft_train.npy, Ysoft_val.npy, and Ysoft_test.npy in TEACHER_PROB_DIR."
        )

    class Top2Gate(layers.Layer):
        def __init__(
            self,
            num_experts=3,
            gate_hidden=128,
            tau_init=2.0,
            lb_coef=0.02,
            ent_coef=0.01,
            name="gate",
            **kwargs,
        ):
            super().__init__(name=name, **kwargs)
            self.num_experts = num_experts
            self.gate_hidden = gate_hidden
            self.tau_init = tau_init
            self.lb_coef = lb_coef
            self.ent_coef = ent_coef
            self.tau = tf.Variable(tau_init, dtype=tf.float32, trainable=False, name=f"{name}_tau")
            self.d1 = layers.Dense(gate_hidden, activation="relu", name=f"{name}_dense1")
            self.d2 = layers.Dense(num_experts, activation=None, name=f"{name}_logits")

        def call(self, gate_in, training=None):
            logits = self.d2(self.d1(gate_in))
            w_soft = tf.nn.softmax(logits / self.tau, axis=-1)

            top2 = tf.math.top_k(w_soft, k=2)
            idx = top2.indices
            vals = tf.nn.softmax(top2.values, axis=-1)

            batch_size = tf.shape(w_soft)[0]
            row_ids = tf.repeat(tf.range(batch_size), repeats=2)
            col_ids = tf.reshape(idx, (-1,))
            scatter_idx = tf.stack([row_ids, col_ids], axis=1)
            scatter_vals = tf.reshape(vals, (-1,))
            w_sparse = tf.scatter_nd(scatter_idx, scatter_vals, (batch_size, self.num_experts))

            mean_usage = tf.reduce_mean(w_sparse, axis=0)
            uniform = tf.fill([self.num_experts], 1.0 / tf.cast(self.num_experts, tf.float32))
            lb_loss = tf.reduce_sum(tf.square(mean_usage - uniform))
            self.add_loss(self.lb_coef * lb_loss)

            ent = -tf.reduce_mean(
                tf.reduce_sum(w_soft * tf.math.log(tf.clip_by_value(w_soft, 1e-8, 1.0)), axis=-1)
            )
            self.add_loss(self.ent_coef * (-ent))

            return w_sparse, w_soft

        def get_config(self):
            config = super().get_config()
            config.update({
                "num_experts": self.num_experts,
                "gate_hidden": self.gate_hidden,
                "tau_init": self.tau_init,
                "lb_coef": self.lb_coef,
                "ent_coef": self.ent_coef,
            })
            return config

    custom_objects = {"Top2Gate": Top2Gate}
    try:
        from tcn import TCN
        custom_objects["TCN"] = TCN
    except Exception as exc:
        print("Could not import TCN. Loading will still work for non-TCN teacher models.")
        print(exc)

    def _teacher_tcn_block(x, filters, kernel_size, dilation_rate, dropout=0.2, name=None):
        res = x
        y = layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            name=None if name is None else name + "_c1",
        )(x)
        y = layers.BatchNormalization(name=None if name is None else name + "_bn1")(y)
        y = layers.ReLU(name=None if name is None else name + "_relu1")(y)
        y = layers.Dropout(dropout, name=None if name is None else name + "_do1")(y)
        y = layers.Conv1D(
            filters,
            kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            name=None if name is None else name + "_c2",
        )(y)
        y = layers.BatchNormalization(name=None if name is None else name + "_bn2")(y)
        if res.shape[-1] != filters:
            res = layers.Conv1D(filters, 1, padding="same", name=None if name is None else name + "_proj")(res)
            res = layers.BatchNormalization(name=None if name is None else name + "_proj_bn")(res)
        y = layers.Add(name=None if name is None else name + "_add")([y, res])
        return layers.ReLU(name=None if name is None else name + "_relu2")(y)

    def _teacher_expert_tcn(x, name_prefix):
        y = _teacher_tcn_block(x, 128, 3, 1, 0.2, name=name_prefix + "_tcn1")
        y = _teacher_tcn_block(y, 128, 3, 2, 0.2, name=name_prefix + "_tcn2")
        y = _teacher_tcn_block(y, 128, 3, 4, 0.2, name=name_prefix + "_tcn3")
        return layers.GlobalAveragePooling1D(name=name_prefix + "_gap")(y)

    def _build_teacher_moe_fallback():
        seq_in = layers.Input(shape=(SEQ_LENGTH, len(AU_NAMES)), name="seq")

        exp_nat = _teacher_expert_tcn(seq_in, "exp_nat")
        exp_act = _teacher_expert_tcn(seq_in, "exp_act")
        exp_shared = _teacher_expert_tcn(seq_in, "exp_shared")

        e1 = layers.Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            output_shape=(1, 128),
            name="exp_nat_expand",
        )(exp_nat)
        e2 = layers.Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            output_shape=(1, 128),
            name="exp_act_expand",
        )(exp_act)
        e3 = layers.Lambda(
            lambda z: tf.expand_dims(z, axis=1),
            output_shape=(1, 128),
            name="exp_share_expand",
        )(exp_shared)
        expert_stack = layers.Concatenate(axis=1, name="experts_stack")([e1, e2, e3])

        gate_in = layers.GlobalAveragePooling1D(name="gate_pool")(seq_in)
        gate_layer = Top2Gate(
            num_experts=3,
            gate_hidden=128,
            tau_init=2.0,
            lb_coef=0.02,
            ent_coef=0.01,
            name="gate",
        )
        w_sparse, _ = gate_layer(gate_in)

        w_expanded = layers.Lambda(
            lambda z: tf.expand_dims(z, -1),
            output_shape=(3, 1),
            name="w_expand",
        )(w_sparse)
        fused = layers.Lambda(
            lambda zs: tf.reduce_sum(zs[0] * zs[1], axis=1),
            output_shape=(128,),
            name="fuse",
        )([w_expanded, expert_stack])

        x = layers.Dense(256, activation="relu", name="clf_dense")(fused)
        x = layers.Dropout(0.3, name="clf_do")(x)
        y_out = layers.Dense(len(le.classes_), activation="softmax", name="y")(x)
        return models.Model(seq_in, y_out, name="MoE_SingleInput")

    def _load_weights_from_keras_archive(model, keras_path):
        try:
            model.load_weights(keras_path)
            return model
        except Exception as direct_exc:
            import tempfile
            import zipfile

            print("Direct load_weights from .keras failed; extracting model.weights.h5.")
            print(direct_exc)

            with tempfile.TemporaryDirectory() as tmp_dir:
                with zipfile.ZipFile(keras_path, "r") as archive:
                    archive.extract("model.weights.h5", tmp_dir)
                model.load_weights(Path(tmp_dir) / "model.weights.h5")
            return model

    try:
        return tf.keras.models.load_model(
            TEACHER_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        try:
            return tf.keras.models.load_model(
                TEACHER_MODEL_PATH,
                custom_objects=custom_objects,
                compile=False,
            )
        except Exception as exc:
            print("Serialized teacher load failed. Rebuilding teacher architecture and loading weights.")
            print(exc)
            teacher_model = _build_teacher_moe_fallback()
            return _load_weights_from_keras_archive(teacher_model, TEACHER_MODEL_PATH)
    except Exception as exc:
        print("Serialized teacher load failed. Rebuilding teacher architecture and loading weights.")
        print(exc)
        teacher_model = _build_teacher_moe_fallback()
        return _load_weights_from_keras_archive(teacher_model, TEACHER_MODEL_PATH)


def teacher_probs_from_model(X_teacher, split_name):
    teacher_model = load_teacher_model()
    probs = teacher_model.predict(X_teacher, batch_size=64, verbose=1)
    if isinstance(probs, (list, tuple)):
        probs = probs[0]

    probs = soften_probabilities(probs, TEACHER_TEMPERATURE)
    TEACHER_PROB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TEACHER_PROB_DIR / f"Ysoft_{split_name}.npy", probs)
    return probs


def load_or_create_teacher_probs(split_name, X_teacher):
    prob_path = TEACHER_PROB_DIR / f"Ysoft_{split_name}.npy"
    if prob_path.exists():
        probs = np.load(prob_path)
        print(f"Loaded {split_name} teacher probabilities:", prob_path, probs.shape)
        return soften_probabilities(probs, TEACHER_TEMPERATURE)

    print(f"Creating {split_name} teacher probabilities from:", TEACHER_MODEL_PATH)
    return teacher_probs_from_model(X_teacher, split_name)


Ysoft_train = load_or_create_teacher_probs("train", X_teacher_train)
Ysoft_val = load_or_create_teacher_probs("val", X_teacher_val)
Ysoft_test = load_or_create_teacher_probs("test", X_teacher_test)

for name, soft, hard in [
    ("train", Ysoft_train, Yemo_train),
    ("val", Ysoft_val, Yemo_val),
    ("test", Ysoft_test, Yemo_test),
]:
    print(name, soft.shape, hard.shape, "row sums", soft.sum(axis=1).min(), soft.sum(axis=1).max())
    assert soft.shape == (len(hard), len(le.classes_))
"""
        ),
        md("## Distillation Losses and Metrics"),
        code(
            """
def soft_categorical_ce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
    return tf.keras.losses.categorical_crossentropy(y_true, y_pred)


def sparse_ce(y_true, y_pred):
    return tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def macro_f1_np(y_true, y_pred, num_classes):
    f1s = []
    for cls in range(num_classes):
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def evaluate_ablation(model, inputs, y_true, name):
    outputs = model.predict(inputs, batch_size=64, verbose=0)
    if isinstance(outputs, (list, tuple)):
        probs = outputs[0]
    elif isinstance(outputs, dict):
        probs = outputs.get("soft_output", next(iter(outputs.values())))
    else:
        probs = outputs

    pred = np.argmax(probs, axis=1)
    acc = float(np.mean(pred == y_true))
    f1 = macro_f1_np(y_true, pred, len(le.classes_))

    print(f"\\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1:.4f}")
    print(classification_report(
        y_true,
        pred,
        labels=np.arange(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0,
    ))
    cm = confusion_matrix(y_true, pred, labels=np.arange(len(le.classes_)))
    display(pd.DataFrame(cm, index=le.classes_, columns=le.classes_))
    return {"accuracy": acc, "macro_f1": f1}
"""
        ),
        md("## Distillation Process 1: Flat Fusion"),
        md(
            """
Flat fusion concatenates eye, head, and hand windows at the input. It has no router or expert routing path, and it is trained only against the teacher soft labels.
"""
        ),
        md("## Distillation Process 2: Coarse MoE"),
        md(
            """
Coarse MoE uses natural, acted, and shared experts with a domain-level gate. It receives the recording domain label as the coarse routing signal and trains with both teacher soft labels and hard emotion labels.
"""
        ),
        md("## Distillation Process 3: EMOE with Unimodal Distillation"),
        md(
            """
EMOE keeps separate eye, head, and hand experts, routes samples conditionally, and adds unimodal distillation losses so each expert's standalone prediction remains calibrated to the teacher distribution.
"""
        ),
        md("## Model Builders"),
        code(
            """
def tcn_expert(input_tensor, name, units=128, dropout=0.2):
    x = layers.Conv1D(units, 3, padding="causal", dilation_rate=1, activation="relu", name=f"{name}_conv1")(input_tensor)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Dropout(dropout, name=f"{name}_drop1")(x)
    x = layers.Conv1D(units, 3, padding="causal", dilation_rate=2, activation="relu", name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Conv1D(units, 3, padding="causal", dilation_rate=4, activation="relu", name=f"{name}_conv3")(x)
    x = layers.GlobalAveragePooling1D(name=f"{name}_pool")(x)
    return layers.Dense(units, activation="relu", name=f"{name}_repr")(x)


def classifier_from_repr(repr_tensor, name, hidden=256, dropout=0.3):
    x = layers.Dense(hidden, activation="relu", name=f"{name}_hidden")(repr_tensor)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.Dense(len(le.classes_), activation="softmax", name=f"{name}_probs")(x)


def duplicate_outputs(main_probs, include_hard=True, prefix=""):
    outputs = {"soft_output": layers.Lambda(lambda z: z, name=f"{prefix}soft_output")(main_probs)}
    if include_hard:
        outputs["hard_output"] = layers.Lambda(lambda z: z, name=f"{prefix}hard_output")(main_probs)
    return outputs


def build_flat_fusion_model():
    flat_input = layers.Input(shape=(SEQ_LENGTH, len(STUDENT_COLS)), name="flat_student_input")
    repr_tensor = tcn_expert(flat_input, "flat_fusion")
    probs = classifier_from_repr(repr_tensor, "flat_fusion_classifier")
    model = models.Model(flat_input, duplicate_outputs(probs, include_hard=False), name="rq2_flat_fusion_soft_only")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss={"soft_output": soft_categorical_ce},
        metrics={"soft_output": ["accuracy"]},
    )
    return model


def build_coarse_domain_moe_model():
    flat_input = layers.Input(shape=(SEQ_LENGTH, len(STUDENT_COLS)), name="flat_student_input")
    domain_input = layers.Input(shape=(), dtype=tf.int32, name="domain_input")

    natural_expert = tcn_expert(flat_input, "natural_domain_expert")
    acted_expert = tcn_expert(flat_input, "acted_domain_expert")
    shared_expert = tcn_expert(flat_input, "shared_domain_expert")

    expert_stack = layers.Concatenate(axis=1, name="domain_expert_stack")([
        layers.Reshape((1, 128), name="natural_expert_axis")(natural_expert),
        layers.Reshape((1, 128), name="acted_expert_axis")(acted_expert),
        layers.Reshape((1, 128), name="shared_expert_axis")(shared_expert),
    ])

    domain_one_hot = layers.Lambda(
        lambda d: tf.one_hot(tf.cast(d, tf.int32), depth=2),
        name="domain_one_hot",
    )(domain_input)
    gate = layers.Dense(16, activation="relu", name="domain_gate_hidden")(domain_one_hot)
    gate = layers.Dense(3, activation="softmax", name="domain_gate")(gate)
    fused = layers.Dot(axes=(1, 1), name="domain_weighted_experts")([gate, expert_stack])
    probs = classifier_from_repr(fused, "coarse_moe_classifier")

    model = models.Model(
        [flat_input, domain_input],
        duplicate_outputs(probs, include_hard=True),
        name="rq2_coarse_domain_moe",
    )
    return model


def build_emoe_unimodal_distillation_model():
    eye_input = layers.Input(shape=(SEQ_LENGTH, len(EYE_COLS)), name="eye_input")
    head_input = layers.Input(shape=(SEQ_LENGTH, len(HEAD_COLS)), name="head_input")
    hand_input = layers.Input(shape=(SEQ_LENGTH, len(HAND_COLS)), name="hand_input")

    eye_repr = tcn_expert(eye_input, "eye_expert")
    head_repr = tcn_expert(head_input, "head_expert")
    hand_repr = tcn_expert(hand_input, "hand_expert")

    expert_stack = layers.Concatenate(axis=1, name="modality_expert_stack")([
        layers.Reshape((1, 128), name="eye_expert_axis")(eye_repr),
        layers.Reshape((1, 128), name="head_expert_axis")(head_repr),
        layers.Reshape((1, 128), name="hand_expert_axis")(hand_repr),
    ])

    router_input = layers.Concatenate(name="sample_conditional_router_input")([eye_repr, head_repr, hand_repr])
    router = layers.Dense(128, activation="relu", name="sample_router_hidden")(router_input)
    router = layers.Dense(3, activation="softmax", name="sample_conditional_router")(router)
    fused = layers.Dot(axes=(1, 1), name="emoe_weighted_experts")([router, expert_stack])

    main_probs = classifier_from_repr(fused, "emoe_classifier")
    outputs = duplicate_outputs(main_probs, include_hard=True)
    outputs["eye_unimodal_output"] = classifier_from_repr(eye_repr, "eye_unimodal")
    outputs["head_unimodal_output"] = classifier_from_repr(head_repr, "head_unimodal")
    outputs["hand_unimodal_output"] = classifier_from_repr(hand_repr, "hand_unimodal")

    return models.Model(
        [eye_input, head_input, hand_input],
        outputs,
        name="rq2_emoe_unimodal_distillation",
    )


flat_model = build_flat_fusion_model()
coarse_model = build_coarse_domain_moe_model()
emoe_model = build_emoe_unimodal_distillation_model()

flat_model.summary()
coarse_model.summary()
emoe_model.summary()
"""
        ),
        md("## Training Helpers"),
        code(
            """
def callbacks_for(name, monitor="val_soft_output_accuracy"):
    ckpt_path = models_path / f"{name}.best.keras"
    return [
        callbacks.EarlyStopping(monitor=monitor, patience=8, restore_best_weights=True),
        callbacks.ModelCheckpoint(ckpt_path, monitor=monitor, save_best_only=True),
    ]


def compile_coarse_or_emoe(model, weights):
    loss = {
        "soft_output": soft_categorical_ce,
        "hard_output": sparse_ce,
    }
    loss_weights = {
        "soft_output": weights["soft"],
        "hard_output": weights["hard"],
    }

    if "eye_unimodal_output" in model.output_names:
        for name in ["eye_unimodal_output", "head_unimodal_output", "hand_unimodal_output"]:
            loss[name] = soft_categorical_ce
            loss_weights[name] = weights["unimodal"]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=loss,
        loss_weights=loss_weights,
        metrics={"soft_output": ["accuracy"], "hard_output": ["accuracy"]},
    )
    return model


def train_flat():
    model = build_flat_fusion_model()
    history = model.fit(
        X_flat_train,
        {"soft_output": Ysoft_train},
        validation_data=(X_flat_val, {"soft_output": Ysoft_val}),
        epochs=30,
        batch_size=32,
        callbacks=callbacks_for("rq2_flat_fusion", monitor="val_soft_output_accuracy"),
        verbose=1,
    )
    return model, history


def train_coarse(weights):
    model = compile_coarse_or_emoe(build_coarse_domain_moe_model(), weights)
    history = model.fit(
        [X_flat_train, Ydom_train],
        {"soft_output": Ysoft_train, "hard_output": Yemo_train},
        validation_data=([X_flat_val, Ydom_val], {"soft_output": Ysoft_val, "hard_output": Yemo_val}),
        epochs=30,
        batch_size=32,
        callbacks=callbacks_for("rq2_coarse_domain_moe", monitor="val_soft_output_accuracy"),
        verbose=1,
    )
    return model, history


def train_emoe(weights):
    model = compile_coarse_or_emoe(build_emoe_unimodal_distillation_model(), weights)
    y_train = {
        "soft_output": Ysoft_train,
        "hard_output": Yemo_train,
        "eye_unimodal_output": Ysoft_train,
        "head_unimodal_output": Ysoft_train,
        "hand_unimodal_output": Ysoft_train,
    }
    y_val = {
        "soft_output": Ysoft_val,
        "hard_output": Yemo_val,
        "eye_unimodal_output": Ysoft_val,
        "head_unimodal_output": Ysoft_val,
        "hand_unimodal_output": Ysoft_val,
    }
    history = model.fit(
        [X_eye_train, X_head_train, X_hand_train],
        y_train,
        validation_data=([X_eye_val, X_head_val, X_hand_val], y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks_for("rq2_emoe_unimodal_distillation", monitor="val_soft_output_accuracy"),
        verbose=1,
    )
    return model, history
"""
        ),
        md("## Run Distillation Process 1: Flat Fusion"),
        md(
            """
This trains the no-routing flat-fusion baseline with soft-label loss only.
"""
        ),
        md("## Run Distillation Processes 2 and 3: Validation-Tuned Joint Losses"),
        md(
            """
This tunes the soft, hard, and unimodal loss weights on the held-out validation participant for Coarse MoE and EMOE.
"""
        ),
        code(
            """
def best_weighted_model(train_fn, eval_inputs, family_name):
    best = None
    trials = []

    for weights in DISTILLATION_WEIGHT_GRID:
        print(f"\\nTraining {family_name} with weights:", weights)
        model, history = train_fn(weights)
        metrics = evaluate_ablation(model, eval_inputs, Yemo_val, f"{family_name} validation")
        trials.append({"weights": weights, "metrics": metrics, "model": model, "history": history})

        if best is None or metrics["macro_f1"] > best["metrics"]["macro_f1"]:
            best = trials[-1]

    print(f"\\nBest {family_name} weights:", best["weights"])
    print("Best validation metrics:", best["metrics"])
    return best, trials


flat_model, flat_history = train_flat()
flat_val_metrics = evaluate_ablation(flat_model, X_flat_val, Yemo_val, "Flat fusion validation")

best_coarse, coarse_trials = best_weighted_model(
    train_coarse,
    [X_flat_val, Ydom_val],
    "Coarse domain MoE",
)

best_emoe, emoe_trials = best_weighted_model(
    train_emoe,
    [X_eye_val, X_head_val, X_hand_val],
    "EMOE + unimodal distillation",
)
"""
        ),
        md("## Held-Out Test Evaluation and Save Models"),
        code(
            """
results = {}
results["flat_fusion"] = evaluate_ablation(flat_model, X_flat_test, Yemo_test, "Flat fusion test")
results["coarse_domain_moe"] = evaluate_ablation(
    best_coarse["model"],
    [X_flat_test, Ydom_test],
    Yemo_test,
    "Coarse domain MoE test",
)
results["emoe_unimodal_distillation"] = evaluate_ablation(
    best_emoe["model"],
    [X_eye_test, X_head_test, X_hand_test],
    Yemo_test,
    "EMOE + unimodal distillation test",
)

results_df = pd.DataFrame(results).T
display(results_df)

def zero_like(x):
    return np.zeros_like(x)


def flat_inputs_with_dropped_modality(modality):
    eye = zero_like(X_eye_test) if modality == "eye" else X_eye_test
    head = zero_like(X_head_test) if modality == "head" else X_head_test
    hand = zero_like(X_hand_test) if modality == "hand" else X_hand_test
    return np.concatenate([eye, head, hand], axis=-1)


def emoe_inputs_with_dropped_modality(modality):
    eye = zero_like(X_eye_test) if modality == "eye" else X_eye_test
    head = zero_like(X_head_test) if modality == "head" else X_head_test
    hand = zero_like(X_hand_test) if modality == "hand" else X_hand_test
    return [eye, head, hand]


ablation_rows = []
for modality in ["eye", "head", "hand"]:
    flat_drop_metrics = evaluate_ablation(
        flat_model,
        flat_inputs_with_dropped_modality(modality),
        Yemo_test,
        f"Flat fusion test without {modality}",
    )
    coarse_drop_metrics = evaluate_ablation(
        best_coarse["model"],
        [flat_inputs_with_dropped_modality(modality), Ydom_test],
        Yemo_test,
        f"Coarse domain MoE test without {modality}",
    )
    emoe_drop_metrics = evaluate_ablation(
        best_emoe["model"],
        emoe_inputs_with_dropped_modality(modality),
        Yemo_test,
        f"EMOE + unimodal distillation test without {modality}",
    )

    for scheme, base_metrics, drop_metrics in [
        ("flat_fusion", results["flat_fusion"], flat_drop_metrics),
        ("coarse_domain_moe", results["coarse_domain_moe"], coarse_drop_metrics),
        ("emoe_unimodal_distillation", results["emoe_unimodal_distillation"], emoe_drop_metrics),
    ]:
        ablation_rows.append({
            "scheme": scheme,
            "dropped_modality": modality,
            "accuracy": drop_metrics["accuracy"],
            "macro_f1": drop_metrics["macro_f1"],
            "accuracy_drop": base_metrics["accuracy"] - drop_metrics["accuracy"],
            "macro_f1_drop": base_metrics["macro_f1"] - drop_metrics["macro_f1"],
        })

modality_ablation_df = pd.DataFrame(ablation_rows)
display(modality_ablation_df.sort_values(["dropped_modality", "scheme"]))

flat_model.save(models_path / "rq2_flat_fusion_soft_only.keras")
best_coarse["model"].save(models_path / "rq2_coarse_domain_moe.keras")
best_emoe["model"].save(models_path / "rq2_emoe_unimodal_distillation.keras")

results_df.to_csv(output_path / "rq2_distillation_ablation_results.csv")
modality_ablation_df.to_csv(output_path / "rq2_modality_ablation_drops.csv", index=False)
print("Saved models and results to:", output_path)
"""
        ),
    ]
)

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nb["metadata"]["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}

OUTPUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(OUTPUT)
