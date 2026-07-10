# Student/Base Model Settings Inventory

Kaynaklar:
- Notebook/script kanıtları: `tools/build_student_moe_notebook.py`, `teacher_matched_cells_to_paste.md`, `rq2_cells_to_paste.md`, `student_teacher_matched_experimental_setup.ipynb`
- Sonuç zip'i: `C:\Users\alime\Downloads\drive-download-20260623T183928Z-3-001.zip`
- Ham çıkarım: `model_settings_extracted_raw.txt`

## Ortak Veri Hazırlama

### İlk student MoE 10Hz pipeline

- Seed: `42` (`random`, `numpy`, `tf`)
- Etiketler: `Anger`, `Disgust`, `Fear`, `Happiness`, `Neutral`, `Sadness`, `Surprise`
- Domain etiketleri: `VideoRecordings=0`, `ActingRecordings=1`
- Split: `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
- Split ataması: fold `0..6` train, fold `7` validation, fold `8..9` test
- Participant exclusivity assert edildi: train/val/test katılımcıları kesişmiyor
- Downsampling: 60Hz -> 10Hz, `DOWNSAMPLE_FACTOR=6`, her 6 satırın ortalaması
- Window length: `SEQ_LENGTH=5`
- Window stride: çakışmasız, `start = 0, 5, 10, ...`
- Kırpma: önce 6'ya bölünebilen satıra kırpılıyor, sonra 5'e bölünebilen 10Hz satıra kırpılıyor
- Missing/NaN handling: numeric coercion, inf -> NaN, `ffill`, `bfill`, kalan `0.0`
- Normalizasyon: bu ilk pipeline'da özel mean-centering/abs normalizasyon görünmüyor
- Student inputları ayrı tensor: eye, head, hand

### Teacher-matched / reviewed student pipeline

- Downsampling yine 60Hz -> 10Hz, `DOWNSAMPLE_FACTOR=6`
- Window length yine `SEQ_LENGTH=5`
- Threshold: çoğu deneyde `0.25`; bir deneyde `0.1`; no-threshold varyantları da var
- Teacher-style windowing:
  - Teacher AU feature sayısı: `60`
  - `TEACHER_WINDOW_THRESHOLD=0.25`
  - `TEACHER_WINDOW_CONSECUTIVE=3`
  - Neutral için pencere kabulü: AU değerlerinin threshold altında/ona eşit olması
  - Diğer emotionlar için pencere kabulü: AU değerlerinden en az birinin threshold üstünde olması
  - 5 frame'lik pencere içinde 3 ardışık kabul frame varsa pencere alınır
  - Pencere kabul edilirse `i += SEQ_LENGTH`, değilse `i += 1`
- Student-threshold windowing tarafında benzer mantık var:
  - `STUDENT_THRESHOLD=0.25`
  - `USE_STUDENT_THRESHOLD=True` yapılan RQ2/reviewed hücreleri var
  - Student signal threshold kolonları üzerinden `abs()` değerleri kullanılıyor
- Bazı reviewed hücrelerde per-recording normalization:
  - 10Hz dataframe için recording mean çıkarılıyor
  - Ardından `.abs()` uygulanıyor
- `no_abs_in_normalization` varyantlarında mean subtraction sonrası `abs()` kaldırılmış/denenmiş görünüyor.

## Model Aileleri

### İlk student MoE

- Model adı: `student_MoE_eye_head_hand`
- Uzmanlar: eye expert, head expert, hand expert, shared expert
- Her expert:
  - `Masking(mask_value=0.0)`
  - `Conv1D(units=64, kernel_size=3, padding="same", activation="relu")`
  - `BatchNormalization`
  - ikinci `Conv1D(64, kernel_size=3, padding="same", activation="relu")`
  - `GlobalAveragePooling1D`
  - `Dropout(0.25)`
  - `Dense(64, relu)`
- Gate:
  - Expert vektörleri concat
  - `Dense(64, relu)`
  - `Dense(4, softmax)`
  - weighted sum
- Classifier:
  - `Dropout(0.3)`
  - `Dense(128, relu)`
  - `Dense(7, softmax)`
- Optimizer: `Adam(learning_rate=1e-3)`
- Loss: `sparse_categorical_crossentropy`
- Metrics: `accuracy`
- Training: `epochs=30`, `batch_size=32`
- Early stopping: monitor `val_accuracy`, patience `8`, restore best weights
- Checkpoint monitor: `val_accuracy`

### Teacher-style single-input MoE

- Model: `teacher-style MoE, 3 experts + Top-2 gate`
- Input: eye + head + hand concatenated
- Common zip-confirmed settings:
  - `seq_length=5`
  - `input_size=30` for full eye+head+hand
  - `input_size=23` for no-hand ablation
  - `num_classes=7`
  - `tau_init=2.0`, `tau_final=0.7`
  - `gate_hidden=128`
  - `classifier_hidden=256`
  - `lb_coef=0.02`
  - `ent_coef=0.01`
- Optimizer:
  - matched/RQ2 notes indicate `Adam(learning_rate=1e-3)` for student MoE variants
  - older teacher base cells also contain SGD teacher baselines (`SGD(lr=0.01, momentum=0.95)`) for Conv1D/TCN teacher models, not the later student matched MoE runs
- Training varied by run:
  - early matched runs: `epochs=30`, `batch_size=10`, early stop `val_accuracy`, patience `8`
  - teacher-window/noabs/adamnoreg runs: `epochs=60`, `batch_size=10`, early stop `val_accuracy`, patience `10`
  - batch ablation: `batch_size=16`
  - val-loss ablation: monitor `val_loss`, patience `6`

### RQ2 distillation models

Distillation section trains three student schemes on the same participant-exclusive split:

- Flat fusion:
  - concat eye/head/hand at input
  - no routing
  - soft-label loss only
- Coarse MoE:
  - natural, acted, shared experts
  - soft-label + hard-label loss
- EMOE with unimodal distillation:
  - sample-conditional modality router
  - standalone expert calibration losses
- Teacher:
  - saved teacher path: `teacher-no-eye-AU-acted-natural-MoE_3experts_2gate_with_10participants_data.keras`
  - teacher AU input: 60 no-eye face AU columns
  - teacher soft-label temperature: `2.0`
- Distillation weight grid:
  - soft `1.0`, hard `0.25`, unimodal `0.10`
  - soft `1.0`, hard `0.50`, unimodal `0.25`
  - soft `1.0`, hard `1.00`, unimodal `0.50`
- RQ2 training summary fields indicate:
  - optimizer `Adam(learning_rate=1e-3)`
  - `epochs=30`
  - batch size stored as `RQ2_BATCH_SIZE`
  - `seq_length=5`
  - `downsample_factor=6`
  - threshold used according to `USE_STUDENT_THRESHOLD`

## Zip Result Runs

Sorted by test accuracy from the zip summaries:

| Run | Acc | Macro F1 | Key settings recovered |
| --- | ---: | ---: | --- |
| `student_teacher_style_moe_0.1_threshold_no_class_weight` | 0.3679 | 0.1250 | threshold `0.1`, no class weight; summary lacks train hyperparams |
| `student_teacher_style_moe_no_threshold_no_class_weight` | 0.3658 | 0.1710 | no threshold, no class weight; summary lacks train hyperparams |
| `student_teacher_style_moe_0.25_threshold_with_class_weight` | 0.2779 | 0.1926 | name says class weight, summary says `class_weight_used=False`; verify notebook if this matters |
| `7train_1val_2test_student-no_hand-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.2735 | 0.1664 | 7/1/2 split, threshold `0.25`, class weight, seq 5, input 23, epochs 60, batch 10, patience 10 |
| `student_teacher_style_moe_no_threshold_with_class_weight` | 0.2733 | 0.2000 | name says class weight, summary says `class_weight_used=False`; verify |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_global_norm` | 0.2532 | 0.1560 | threshold `0.25`, global norm, summary says no class weight |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_normalized_per_recording_9train_1test` | 0.2500 | 0.2520 | 9 train / 1 test, per-recording norm, threshold `0.25`; summary says no class weight |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-no_class_weight-MoE_3experts_2gate_together` | 0.2247 | 0.1487 | 7/1/2 by name; threshold `0.25`, no class weight; summary lacks detailed train params |
| `student_teacher_style_moe_0.25_threshold_except_neutral_with_class_weight_global_norm` | 0.2237 | 0.1712 | threshold `0.25` except neutral, global norm, summary says no class weight |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-sqr_root_class_weight-MoE_3experts_2gate_together` | 0.2213 | 0.1704 | name says sqrt class weight but summary says `class_weight_used=False`; verify |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-no_class_weight-MoE_3experts_2gate_together` | 0.2030 | 0.0977 | summary unexpectedly says `class_weight_used=True`; name says no class weight; verify |
| `7train_1val_2test_student-same_hyperparameters_with_teacher_but_batch16-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.2003 | 0.1971 | 7/1/2, threshold `0.25`, class weight, seq 5, input 30, epochs 30, batch 16, patience 8 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-MoE_3experts_2gate_together` | 0.1908 | 0.2571 | threshold `0.25`, no class weight; highest macro F1 among listed summaries |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-class_weight-monitor_val_loss-patience_6-MoE_3experts_2gate_together` | 0.1772 | 0.2067 | 7/1/2, threshold `0.25`, class weight, seq 5, input 30, epochs 30, batch 10, monitor `val_loss`, patience 6 |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized_but_no_abs-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.1688 | 0.1245 | 7/1/2, threshold `0.25`, class weight, seq 5, input 30, epochs 30, batch 10, no abs normalization |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.1613 | 0.0816 | 7/1/2, threshold `0.25`, class weight, seq 5, input 30, epochs 60, batch 10, no abs, no gate reg by name |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.1506 | 0.1486 | 7/1/2, threshold `0.25`, class weight, seq 5, input 30, epochs 30, batch 10 |
| `6train_2val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.1496 | 0.0729 | 6/2/2, threshold `0.25`, class weight, seq 5, input 30, epochs 60, batch 10 |
| `separate_video_acting_student_moe_0.25_threshold_with_class_weight_normalized_per_recording` video | 0.1260 | 0.1117 | separate domain training/eval, threshold `0.25`, class weight, per-recording norm |
| `no_eyeGazePitch_Roll_OriginX_headPitch_Roll_student_teacher_style_moe_0.25_threshold_with_class_weight_normalized_per_recording` | 0.1225 | 0.1313 | removed selected eye/head features, threshold `0.25`, per-recording norm; summary says no class weight |
| `separate_video_acting_student_moe_0.25_threshold_with_class_weight_normalized_per_recording` acting | 0.1197 | 0.0884 | separate domain training/eval, threshold `0.25`, class weight, per-recording norm |

## Known Inconsistencies To Verify

- Several run names contain `with_class_weight`, but their `summary.json` has `class_weight_used: False`.
- One run named `no_class_weight` has `class_weight_used: True` in `summary.json`.
- Older summary files often omit `epochs`, `batch_size`, exact participant lists, and window counts; later summaries contain these fields.
- `input` for the no-hand ablation still says `eye + head + hand concatenated` in summary, but `input_size=23` and run/checkpoint name indicate hand features were removed.

## Most Likely Reconstruction

- The first base student model was the separate-input 4-expert MoE trained on 10Hz non-overlapping windows, 7/1/2 participant-exclusive split, `Adam(1e-3)`, `epochs=30`, `batch_size=32`.
- The later "student_teacher_style_moe" experiments were rewritten to match the teacher more closely:
  - single concatenated input,
  - 3 TCN/temporal experts,
  - sparse Top-2 gate,
  - threshold-based teacher/student style window selection,
  - mostly `seq_length=5`,
  - mostly 7/1/2 split,
  - mostly `batch_size=10`,
  - class-weight and normalization variants.
- Best test accuracy in the zip is from no/low-threshold no-class-weight runs, but best macro-F1 among the listed zip summaries is the `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-MoE_3experts_2gate_together` run.
