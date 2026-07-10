# Zip Run Settings Reconstructed

Zip: `C:\Users\alime\Downloads\drive-download-20260623T183928Z-3-001.zip`

Bu dosya sadece zip içindeki run klasörleri içindir. Ayarları şu sırayla çıkardım:

1. Run klasöründeki `summary.json`, `classification_report.json`, `Yemo_test.npy`
2. Run adı
3. Geçmiş notebook/script hücreleri:
   - `student_MoE_eye_head_hand_10participants_teacher_moe_adapted.ipynb`
   - `student_MoE_eye_head_hand_10participants_thresholded_reviewed.ipynb`
   - `student_MoE_eye_head_hand_10participants_top2_moe_reviewed.ipynb`
   - `student_MoE_eye_head_hand_10participants_student_threshold_reviewed.ipynb`
   - `student_MoE_eye_head_hand_10participants_17_with_rq2.ipynb`
   - `student_teacher_matched_experimental_setup.ipynb`
   - `student_MoE_eye_head_hand_10participants_and_rq2_distillation_continued.ipynb`
4. Previous Codex conversations:
   - `Train Quest 3 student model`
   - `Align datasets and retrain`
   - `Add separate video and acting models`
   - `Reduce EMOE model capacity`

`summary.json` eksik alanlarda kesin konuşmuyorum; `source` sütununda `summary` veya `inferred` olarak belirttim.

## Previous Conversation Evidence

Önceki konuşmaları da kontrol ettim. Bunlar notebooklardan bağımsız ek kanıt veriyor:

- İlk base student model konuşmasında açıkça şu ayarlar vardı: Quest-3 sparse signals only, yani eye + head + hand/controller; `60 FPS -> average every 6 rows -> 10 Hz`; `SEQ_LENGTH = 5`; non-overlapping windows `0-4`, `5-9`, `10-14`; split before windowing; participant-exclusive `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)`; folds `0..6` train, fold `7` validation, folds `8..9` test; training `epochs=30`, `batch_size=32`; three-input base student MoE with eye/head/hand/shared experts and a gate over 4 experts.
- Student-threshold konuşmasında: `USE_STUDENT_THRESHOLD=False` ise threshold tamamen atlanıyor ve doğrudan `range(0, usable_rows, seq_length)` ile non-overlap windows alınıyor. `USE_STUDENT_THRESHOLD=True` ise `abs()` student signal değerleri threshold ile karşılaştırılıyor; Neutral için tüm threshold kolonları `<= 0.25`, non-neutral için en az bir kolon `> 0.25`; 5-frame window içinde 3 ardışık valid frame aranıyor.
- Teacher threshold konuşmasında teacher tarafı netti: non-neutral emotionlar için face AU activity `> 0.25`, Neutral için all face AUs `<= 0.25`, `SEQ_LENGTH=5`, `consecutive=3`.
- Class weight konuşmasında amaç: az örnekli sınıflara daha yüksek loss weight vermek ve training sırasında `model.fit(..., class_weight=class_weight)` kullanmak.
- Separate video/acting konuşması zip'teki klasörle birebir eşleşiyor: base run name `separate_video_acting_student_moe_0.25_threshold_with_class_weight_normalized_per_recording`, altında ayrı `video/` ve `acting/` sonuç klasörleri; common architecture ama video ve acting için bağımsız model ağırlıkları.
- Reduced EMOE konuşması zip kimliklerini doğruluyor: stride-2 zip train 4904 / val 352 / test 2261, original stride-5/non-overlap zip train 2073 / val 144 / test 936. Bu zip'teki 936-window teacher-window runları original/non-overlap aileyle uyumlu.

## Ortak Varsayılanlar

Zip'teki bütün runlar `teacher-style MoE, 3 experts + Top-2 gate` ailesinde görünüyor. Eski minimal summary'lerde detaylar yazılmamış olsa da notebooklardan ortak temel şu:

- `SEQ_LENGTH = 5`
- `DOWNSAMPLE_FACTOR = 6`, yani 60Hz -> 10Hz
- Emotion classes: 7 sınıf
- Input: eye + head + hand concat, bazı ablationlarda feature çıkarılmış
- Top-2 gate:
  - `num_experts=3`
  - `gate_hidden=128`
  - `tau_init=2.0`
  - `tau_final=0.7`
  - `classifier_hidden=256`
  - `lb_coef=0.02`
  - `ent_coef=0.01`
- Later matched training default:
  - `epochs=30`
  - `batch_size=10`
  - early stopping `monitor="val_accuracy"`, `patience=8`
  - checkpoint `monitor="val_accuracy"`
- Teacher-window/no-abs/adamnoreg training default:
  - `epochs=60`
  - `batch_size=10`
  - early stopping `monitor="val_accuracy"`, `patience=10`
  - optimizer `Adam(learning_rate=1e-3)`
- Split default:
  - `StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)`
  - normal split: folds `0..6` train, `7` val, `8..9` test
  - some old/special runs used 9 train / 1 test or 6/2/2, as run name/summary says

## Windowing Variants

- `student_teacher_style_moe_*` early runs:
  - student-signal thresholding from reviewed notebooks
  - threshold columns: `STUDENT_COLS` except `RightHandTracked`
  - non-neutral: keep if any continuous student signal has `abs(value) > threshold`
  - neutral: keep if all continuous student signals have `abs(value) <= threshold`
  - 5-frame window accepted if threshold condition appears as the notebook's consecutive logic
  - many variants used per-recording mean centering followed by `.abs()`
- `teacher_window-*` runs:
  - teacher AU window acceptance using 60 AU columns
  - threshold `0.25`, consecutive `3`
  - neutral: all AU values <= threshold
  - non-neutral: any AU value > threshold
  - accepted window then advances by `SEQ_LENGTH`
- `no_abs_in_normalization`:
  - mean-centering kept, `.abs()` removed
- `same_window_size_with_teacher`:
  - test windows are 936, same as teacher-window selection
- `same_hyperparameters_with_teacher` without same-window-size:
  - test windows are 1473, so windowing differs from the strict teacher-window version

## Per Run Table

| Zip run | Source | Split/window evidence | Main settings |
| --- | --- | --- | --- |
| `6train_2val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | summary | 6 train / 2 val / 2 test participants; windows 1922 / 295 / 936 | threshold 0.25; class weight true; seq 5; input 30; epochs 60; batch 10; monitor val_accuracy; patience 10; Adam 1e-3 inferred from matched cell; no abs normalization; gate reg disabled by name |
| `7train_1val_2test_student-no_hand-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 2073 / 144 / 936 | threshold 0.25; class weight true; seq 5; input 23; epochs 60; batch 10; monitor val_accuracy; patience 10; no hand ablation; no abs normalization; no gate reg by name |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 2073 / 144 / 936 | threshold 0.25; class weight true; seq 5; input 30; epochs 30; batch 10; monitor val_accuracy; patience 8; teacher same window size |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized_but_no_abs-0.25-threshold-class_weight-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 2073 / 144 / 936 | threshold 0.25; class weight true; seq 5; input 30; epochs 30; batch 10; monitor val_accuracy; patience 8; normalized but no abs |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-MoE_3experts_2gate_together` | summary + inferred | Yemo_test has 1473 windows; run name says 7 / 1 / 2 | threshold 0.25; class weight false; likely seq 5, input 30, epochs 30, batch 10, monitor val_accuracy, patience 8; per-recording normalized + abs |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-class_weight-monitor_val_loss-patience_6-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 5100 / 726 / 1473 | threshold 0.25; class weight true; seq 5; input 30; epochs 30; batch 10; monitor val_loss; patience 6; normalized + abs |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-no_class_weight-MoE_3experts_2gate_together` | summary + inferred | Yemo_test has 1473 windows; run name says 7 / 1 / 2 | threshold 0.25; class weight false; likely seq 5, input 30, epochs 30, batch 10, monitor val_accuracy, patience 8; normalized + abs |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-sqr_root_class_weight-MoE_3experts_2gate_together` | summary + notebook | Yemo_test has 1473 windows; sqrt class-weight code exists in matched notebooks | threshold 0.25; intended sqrt class weights; summary says `class_weight_used=False`; likely seq 5, input 30, epochs 30, batch 10, monitor val_accuracy, patience 8 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher_but_batch16-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 5100 / 726 / 1473 | threshold 0.25; class weight true; seq 5; input 30; epochs 30; batch 16; monitor val_accuracy; patience 8; normalized + abs |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | summary | 7 / 1 / 2 participants; windows 2073 / 144 / 936 | threshold 0.25; class weight true; seq 5; input 30; epochs 60; batch 10; monitor val_accuracy; patience 10; Adam 1e-3; no abs; no gate reg by name |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-no_class_weight-MoE_3experts_2gate_together` | summary conflict | 7 / 1 / 2 participants; windows 2073 / 144 / 936 | run name says no class weight, but summary says class weight true with balanced weights; seq 5; input 30; epochs 60; batch 10; monitor val_accuracy; patience 10; Adam 1e-3; no abs; no gate reg |
| `no_eyeGazePitch_Roll_OriginX_headPitch_Roll_student_teacher_style_moe_0.25_threshold_with_class_weight_normalized_per_recording` | summary + inferred | Yemo_test has 1453 windows | threshold 0.25; removed `EyeGazePitch`, `EyeGazeRoll`, `EyeGazeOriginX`, `HeadPitch`, `HeadRoll`; per-recording normalization; summary says class weight false despite run name; likely epochs 30, batch 10, monitor val_accuracy, patience 8 |
| `separate_video_acting_student_moe_0.25_threshold_with_class_weight_normalized_per_recording/acting` | summary + notebook | train participants 7, val 2, test 1; acting test windows 117 | threshold 0.25; class weight true; per-recording normalization; separate domain model; likely seq 5, epochs 30, batch 10, monitor val_accuracy, patience 8 |
| `separate_video_acting_student_moe_0.25_threshold_with_class_weight_normalized_per_recording/video` | summary + notebook | train participants 7, val 2, test 1; video test windows 619 | threshold 0.25; class weight true; per-recording normalization; separate domain model; likely seq 5, epochs 30, batch 10, monitor val_accuracy, patience 8 |
| `student_teacher_style_moe_0.1_threshold_no_class_weight` | summary + inferred | Yemo_test has 1465 windows; no neutral class in test array | threshold 0.1; no class weight; likely 7 / 1 / 2 split; seq 5; downsample 6; teacher-style top2 MoE; likely epochs 30, batch 10 or 32 depending early notebook branch; exact train hyperparams were not saved |
| `student_teacher_style_moe_0.25_threshold_except_neutral_with_class_weight_global_norm` | summary + inferred | Yemo_test has 1605 windows | threshold 0.25 except neutral; global normalization; run name says with class weight but summary says false; likely seq 5; downsample 6; top2 MoE; exact train hyperparams not saved |
| `student_teacher_style_moe_0.25_threshold_with_class_weight` | summary + inferred | Yemo_test has 1605 windows | threshold 0.25; run name says with class weight but summary says false; likely no global/per-recording normalization tag; seq 5; downsample 6; top2 MoE; exact train hyperparams not saved |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_global_norm` | summary + inferred | Yemo_test has 1465 windows; no neutral class in test array | threshold 0.25; global normalization; run name says with class weight but summary says false; likely seq 5; downsample 6; top2 MoE; exact train hyperparams not saved |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_normalized_per_recording_9train_1test` | summary + inferred | 9 train / 1 test by run name; Yemo_test has 736 windows | threshold 0.25; per-recording normalization; run name says class weight but summary says false; likely no validation split or validation folded into training depending notebook cell; exact train hyperparams not saved |
| `student_teacher_style_moe_no_threshold_no_class_weight` | summary + inferred | Yemo_test has 1621 windows | no threshold; no class weight; likely seq 5; downsample 6; top2 MoE; exact train hyperparams not saved |
| `student_teacher_style_moe_no_threshold_with_class_weight` | summary + inferred | Yemo_test has 1621 windows | no threshold; run name says with class weight but summary says false; likely seq 5; downsample 6; top2 MoE; exact train hyperparams not saved |

## Scores From Zip

| Run | Accuracy | Macro F1 | Test windows |
| --- | ---: | ---: | ---: |
| `student_teacher_style_moe_0.1_threshold_no_class_weight` | 0.3679 | 0.1250 | 1465 |
| `student_teacher_style_moe_no_threshold_no_class_weight` | 0.3658 | 0.1710 | 1621 |
| `student_teacher_style_moe_0.25_threshold_with_class_weight` | 0.2779 | 0.1926 | 1605 |
| `7train_1val_2test_student-no_hand-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.2735 | 0.1664 | 936 |
| `student_teacher_style_moe_no_threshold_with_class_weight` | 0.2733 | 0.2000 | 1621 |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_global_norm` | 0.2532 | 0.1560 | 1465 |
| `student_teacher_style_moe_0.25_threshold_with_class_weight_normalized_per_recording_9train_1test` | 0.2500 | 0.2520 | 736 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-no_class_weight-MoE_3experts_2gate_together` | 0.2247 | 0.1487 | 1473 |
| `student_teacher_style_moe_0.25_threshold_except_neutral_with_class_weight_global_norm` | 0.2237 | 0.1712 | 1605 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-sqr_root_class_weight-MoE_3experts_2gate_together` | 0.2213 | 0.1704 | 1473 |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-no_class_weight-MoE_3experts_2gate_together` | 0.2030 | 0.0977 | 936 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher_but_batch16-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.2003 | 0.1971 | 1473 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-MoE_3experts_2gate_together` | 0.1908 | 0.2571 | 1473 |
| `7train_1val_2test_student-same_hyperparameters_with_teacher-normalized-0.25-threshold-class_weight-monitor_val_loss-patience_6-MoE_3experts_2gate_together` | 0.1772 | 0.2067 | 1473 |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized_but_no_abs-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.1688 | 0.1245 | 936 |
| `7train_1val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.1613 | 0.0816 | 936 |
| `7train_1val_2test_student-same_hyperparameters_and_same_window_size_with_teacher-normalized-0.25-threshold-class_weight-MoE_3experts_2gate_together` | 0.1506 | 0.1486 | 936 |
| `6train_2val_2test_student-teacher_window-no_abs_in_normalization-no_gate_reg-adam-class_weight-MoE_3experts_2gate_together` | 0.1496 | 0.0729 | 936 |
| `separate_video_acting.../video` | 0.1260 | 0.1117 | 619 |
| `no_eyeGazePitch_Roll_OriginX_headPitch_Roll...` | 0.1225 | 0.1313 | 1453 |
| `separate_video_acting.../acting` | 0.1197 | 0.0884 | 117 |

## Hala Belirsiz Kalanlar

- Eski `student_teacher_style_moe_*` runlarında `epochs`, `batch_size`, optimizer ve exact split participants kaydedilmemiş. Notebook geçmişine göre en olası ayarlar `seq=5`, `downsample=6`, Top-2 MoE, `val_accuracy` early stopping, ama batch `32` olan erken hücreler ve batch `10` olan matched hücreler ikisi de geçmişte var.
- `with_class_weight` geçen bazı runlarda summary `class_weight_used=False` diyor. Bu nedenle raporda iki kanıtı ayrı tuttum.
- `sqr_root_class_weight` runında notebookta sqrt class weight kodu var, ama summary yine `False` diyor. Bu da kayıt bug'ı veya yanlış summary overwrite olabilir.
- `no_class_weight` isimli teacher-window runında summary `class_weight_used=True` diyor. Burada summary daha güçlü kanıt, ama run adıyla çelişiyor.
