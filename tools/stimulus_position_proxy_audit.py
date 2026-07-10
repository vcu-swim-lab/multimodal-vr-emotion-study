from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


GAZE_HEAD_COLS = [
    "EyeGazeOriginX",
    "EyeGazeOriginY",
    "EyeGazeOriginZ",
    "EyeGazeDirX",
    "EyeGazeDirY",
    "EyeGazeDirZ",
    "EyeGazeYaw",
    "EyeGazePitch",
    "EyeGazeRoll",
    "EyeGazeRotX",
    "EyeGazeRotY",
    "EyeGazeRotZ",
    "EyeGazeRotW",
    "HeadYaw",
    "HeadPitch",
]

META_COLS = ["Timestamp", "RealtimeSeconds", "Emotion"]
ANGLE_COLS = {"EyeGazeYaw", "EyeGazePitch", "EyeGazeRoll", "HeadYaw", "HeadPitch"}


def wrap_degrees(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return ((arr + 180.0) % 360.0) - 180.0


def circular_mean_degrees(values: pd.Series | np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    sin_mean = np.nanmean(np.sin(radians))
    cos_mean = np.nanmean(np.cos(radians))
    return float(wrap_degrees(np.rad2deg(np.arctan2(sin_mean, cos_mean))))


def discover_recordings(data_root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted((data_root / "ActingRecordings").glob("*/*/acting.csv")):
        rows.append(parse_path(path, "ActingRecordings"))
    for path in sorted((data_root / "VideoRecordings").glob("*/*/weights.csv")):
        rows.append(parse_path(path, "VideoRecordings"))
    if not rows:
        raise FileNotFoundError(f"No acting/video CSV recordings found under {data_root}")
    return pd.DataFrame(rows)


def parse_path(path: Path, recording_type: str) -> dict:
    return {
        "path": path,
        "recording_type": recording_type,
        "source": "acting" if recording_type == "ActingRecordings" else "video",
        "participant": path.parents[1].name,
        "emotion": path.parent.name,
    }


def downsample_60hz_to_10hz(df: pd.DataFrame, factor: int = 6) -> pd.DataFrame:
    usable_rows = (len(df) // factor) * factor
    if usable_rows == 0:
        return df.iloc[0:0].copy()
    df = df.iloc[:usable_rows].reset_index(drop=True)
    return df.groupby(np.arange(len(df)) // factor).mean(numeric_only=True).reset_index(drop=True)


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    return numeric.ffill().bfill().fillna(0.0)


def direction_bin(value: float, neg_label: str, center_label: str, pos_label: str, threshold: float = 5.0) -> str:
    if value < -threshold:
        return neg_label
    if value > threshold:
        return pos_label
    return center_label


def summarize_recording(row: pd.Series, downsample: bool) -> dict:
    path = Path(row["path"])
    header = pd.read_csv(path, nrows=0).columns.tolist()
    available = [c for c in GAZE_HEAD_COLS if c in header]
    missing = sorted(set(GAZE_HEAD_COLS) - set(available))
    if not available:
        raise ValueError(f"No gaze/head audit columns found in {path}")

    raw = pd.read_csv(path, usecols=available)
    values = clean_numeric(raw)
    if downsample:
        values = downsample_60hz_to_10hz(values)

    summary = {
        "path": str(path),
        "source": row["source"],
        "recording_type": row["recording_type"],
        "participant": row["participant"],
        "emotion": row["emotion"],
        "n_frames_or_samples": int(len(values)),
        "missing_audit_cols": ";".join(missing),
    }
    for col in available:
        if col in ANGLE_COLS:
            wrapped = pd.Series(wrap_degrees(values[col]))
            summary[f"{col}_mean"] = circular_mean_degrees(values[col])
            summary[f"{col}_median"] = float(wrapped.median())
            summary[f"{col}_std"] = float(wrapped.std(ddof=0))
            summary[f"{col}_iqr"] = float(wrapped.quantile(0.75) - wrapped.quantile(0.25))
            continue

        summary[f"{col}_mean"] = float(values[col].mean())
        summary[f"{col}_median"] = float(values[col].median())
        summary[f"{col}_std"] = float(values[col].std(ddof=0))
        summary[f"{col}_iqr"] = float(values[col].quantile(0.75) - values[col].quantile(0.25))

    yaw_mean = summary.get("EyeGazeYaw_mean", np.nan)
    pitch_mean = summary.get("EyeGazePitch_mean", np.nan)
    summary["yaw_direction_bin"] = direction_bin(yaw_mean, "left", "center", "right")
    summary["pitch_direction_bin"] = direction_bin(pitch_mean, "down", "center", "up")
    return summary


def grouped_numeric_summary(recording_summary: pd.DataFrame) -> pd.DataFrame:
    stat_cols = [
        c
        for c in recording_summary.columns
        if c.endswith("_mean") or c.endswith("_median") or c.endswith("_std") or c.endswith("_iqr")
    ]
    grouped = (
        recording_summary
        .groupby(["source", "emotion"], dropna=False)
        .agg(
            n_recordings=("path", "count"),
            n_participants=("participant", "nunique"),
            total_samples=("n_frames_or_samples", "sum"),
            **{f"{col}_avg_across_recordings": (col, "mean") for col in stat_cols},
        )
        .reset_index()
    )
    return grouped


def grouped_direction_counts(recording_summary: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    counts = (
        recording_summary
        .groupby(["source", "emotion", bin_col], dropna=False)
        .size()
        .rename("n_recordings")
        .reset_index()
    )
    totals = counts.groupby(["source", "emotion"])["n_recordings"].transform("sum")
    counts["share"] = counts["n_recordings"] / totals
    return counts.sort_values(["source", "emotion", "share"], ascending=[True, True, False])


def source_range_flags(grouped: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, source_df in grouped.groupby("source"):
        yaw_col = "EyeGazeYaw_mean_avg_across_recordings"
        pitch_col = "EyeGazePitch_mean_avg_across_recordings"
        yaw_range = float(source_df[yaw_col].max() - source_df[yaw_col].min()) if yaw_col in source_df else np.nan
        pitch_range = float(source_df[pitch_col].max() - source_df[pitch_col].min()) if pitch_col in source_df else np.nan
        rows.append(
            {
                "source": source,
                "emotion_count": int(source_df["emotion"].nunique()),
                "eye_gaze_yaw_mean_range_degrees": yaw_range,
                "eye_gaze_pitch_mean_range_degrees": pitch_range,
                "flag": bool((yaw_range >= 15.0) or (pitch_range >= 10.0)),
            }
        )
    return pd.DataFrame(rows)


def dominant_bin_flags(counts: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    top = counts.sort_values("share", ascending=False).groupby(["source", "emotion"]).head(1).copy()
    top = top.rename(columns={bin_col: "dominant_bin", "share": "dominant_share"})
    top["flag"] = top["dominant_share"] >= 0.70
    return top[["source", "emotion", "dominant_bin", "dominant_share", "n_recordings", "flag"]]


def markdown_table(df: pd.DataFrame) -> str:
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    display_df = display_df.astype(str)
    headers = list(display_df.columns)
    rows = display_df.values.tolist()
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_report(
    output_dir: Path,
    records: pd.DataFrame,
    grouped: pd.DataFrame,
    yaw_flags: pd.DataFrame,
    pitch_flags: pd.DataFrame,
    range_flags: pd.DataFrame,
) -> None:
    report_path = output_dir / "stimulus_position_proxy_audit_report.md"
    yaw_cols = [
        "source",
        "emotion",
        "n_recordings",
        "n_participants",
        "EyeGazeYaw_mean_avg_across_recordings",
        "EyeGazePitch_mean_avg_across_recordings",
        "EyeGazeRoll_mean_avg_across_recordings",
        "EyeGazeOriginX_mean_avg_across_recordings",
        "EyeGazeOriginY_mean_avg_across_recordings",
        "EyeGazeOriginZ_mean_avg_across_recordings",
        "EyeGazeDirX_mean_avg_across_recordings",
        "EyeGazeDirY_mean_avg_across_recordings",
        "EyeGazeDirZ_mean_avg_across_recordings",
        "EyeGazeRotX_mean_avg_across_recordings",
        "EyeGazeRotY_mean_avg_across_recordings",
        "EyeGazeRotZ_mean_avg_across_recordings",
        "EyeGazeRotW_mean_avg_across_recordings",
        "HeadYaw_mean_avg_across_recordings",
        "HeadPitch_mean_avg_across_recordings",
    ]
    yaw_cols = [c for c in yaw_cols if c in grouped.columns]

    lines = [
        "# Stimulus-Position Proxy Audit",
        "",
        "This audit checks whether emotion labels are systematically associated with gaze/head direction.",
        "The CSV files do not contain explicit stimulus-position metadata, so this is a proxy audit using gaze and head direction signals.",
        "",
        "## Data Scanned",
        "",
        f"- Recordings: {len(records)}",
        f"- Sources: {', '.join(sorted(records['source'].unique()))}",
        f"- Participants: {records['participant'].nunique()}",
        f"- Emotions: {', '.join(sorted(records['emotion'].unique()))}",
        "",
        "## Main Direction Summary",
        "",
        markdown_table(grouped[yaw_cols].sort_values(["source", "emotion"])),
        "",
        "## Dominant Mean-Yaw Bins",
        "",
        "A dominant share near 1.0 means most recordings for that source/emotion have the same average left/center/right gaze direction.",
        "",
        markdown_table(yaw_flags.sort_values(["source", "emotion"])),
        "",
        "## Dominant Mean-Pitch Bins",
        "",
        markdown_table(pitch_flags.sort_values(["source", "emotion"])),
        "",
        "## Across-Emotion Direction Range By Source",
        "",
        "Large ranges suggest that emotion classes differ in gaze direction. This can be emotional behavior, stimulus-position confounding, or both.",
        "",
        markdown_table(range_flags),
        "",
        "## Interpretation Guide",
        "",
        "- If each emotion/source has a different dominant yaw or pitch bin, the eye model may partly be learning where the stimulus appeared.",
        "- If video and acting have different direction patterns, a cross-source eye-only test becomes especially important.",
        "- This audit does not prove the confound by itself because explicit stimulus positions are not logged here.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy audit for stimulus-position confounding in gaze/head features.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("multimodal-vr-emotion-study-main"),
        help="Root containing ActingRecordings and VideoRecordings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analyzing-data") / "stimulus_position_proxy_audit",
        help="Directory for audit outputs.",
    )
    parser.add_argument(
        "--no-downsample",
        action="store_true",
        help="Use raw frame rows instead of 60Hz-to-10Hz averaged samples.",
    )
    args = parser.parse_args()

    records = discover_recordings(args.data_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = [
        summarize_recording(row, downsample=not args.no_downsample)
        for _, row in records.iterrows()
    ]
    recording_summary = pd.DataFrame(summaries)
    grouped = grouped_numeric_summary(recording_summary)
    yaw_counts = grouped_direction_counts(recording_summary, "yaw_direction_bin")
    pitch_counts = grouped_direction_counts(recording_summary, "pitch_direction_bin")
    yaw_flags = dominant_bin_flags(yaw_counts, "yaw_direction_bin")
    pitch_flags = dominant_bin_flags(pitch_counts, "pitch_direction_bin")
    ranges = source_range_flags(grouped)

    records.assign(path=records["path"].astype(str)).to_csv(args.output_dir / "recordings_scanned.csv", index=False)
    recording_summary.to_csv(args.output_dir / "recording_level_direction_summary.csv", index=False)
    grouped.to_csv(args.output_dir / "emotion_source_direction_summary.csv", index=False)
    yaw_counts.to_csv(args.output_dir / "emotion_source_yaw_bin_counts.csv", index=False)
    pitch_counts.to_csv(args.output_dir / "emotion_source_pitch_bin_counts.csv", index=False)
    yaw_flags.to_csv(args.output_dir / "dominant_yaw_bin_flags.csv", index=False)
    pitch_flags.to_csv(args.output_dir / "dominant_pitch_bin_flags.csv", index=False)
    ranges.to_csv(args.output_dir / "direction_range_flags.csv", index=False)
    write_report(args.output_dir, records, grouped, yaw_flags, pitch_flags, ranges)

    print(f"Scanned {len(records)} recordings.")
    print(f"Wrote audit outputs to: {args.output_dir.resolve()}")
    print(ranges.to_string(index=False))


if __name__ == "__main__":
    main()
