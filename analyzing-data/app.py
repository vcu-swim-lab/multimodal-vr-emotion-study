import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import re

_DEFAULT_ROOT = Path(__file__).parent.parent / "multimodal-vr-emotion-study-main"
_DEFAULT_FORM_DIR = Path(__file__).parent / "FormResponses"
SIGNAL_PREFIXES = ["Head", "LeftHand", "RightHand", "LeftController", "RightController"]
PHASE_ORDER = ["Video", "Acting"]
EMOTION_LABELS = ["Disgust", "Happiness", "Anger", "Sadness", "Fear", "Neutral", "Surprise"]
HEAD_ROTATION_COLS = ["HeadPitch", "HeadYaw", "HeadRoll"]
HEAD_QUATERNION_COLS = ["HeadRotX", "HeadRotY", "HeadRotZ", "HeadRotW"]
EYE_GAZE_COLS = [
    "EyeGazeTracked",
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


def _discover_sessions(project_root: Path) -> list[str]:
    sessions: set[str] = set()
    for base_name, file_name in (("VideoRecordings", "weights.csv"), ("ActingRecordings", "acting.csv")):
        base = project_root / base_name
        if not base.exists():
            continue
        for session_dir in base.iterdir():
            if session_dir.is_dir() and any(session_dir.glob(f"*/{file_name}")):
                sessions.add(session_dir.name)
    return sorted(sessions)


def _parse_score(value) -> float:
    if pd.isna(value):
        return np.nan
    match = re.match(r"\s*(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan


def _session_datetime(session_id: str) -> pd.Timestamp | None:
    try:
        return pd.to_datetime(session_id, format="%Y%m%d-%H%M")
    except ValueError:
        return None


def _timestamp_column(columns) -> str | None:
    return next((column for column in columns if str(column).strip().lower() == "timestamp"), None)


def _parse_form_timestamps(values: pd.Series) -> pd.Series:
    cleaned = values.astype(str).str.replace(r"\s+[A-Z]{2,4}\s*$", "", regex=True)
    return pd.to_datetime(cleaned, errors="coerce")


def _select_response_rows(raw: pd.DataFrame, session_id: str) -> pd.DataFrame:
    timestamp_col = _timestamp_column(raw.columns)
    session_time = _session_datetime(session_id)
    if timestamp_col is None or session_time is None or raw.empty:
        selected = raw.copy()
        selected["MatchWarning"] = "Could not match by timestamp because Timestamp or session time is missing."
        return selected

    response_times = _parse_form_timestamps(raw[timestamp_col])
    if response_times.notna().sum() == 0:
        selected = raw.copy()
        selected["MatchWarning"] = "No parseable form timestamps found."
        return selected

    same_date = response_times.dt.date == session_time.date()
    after_start = same_date & (response_times >= session_time)
    match_warning = ""

    if after_start.any():
        candidates = response_times[after_start]
        nearest_index = (candidates - session_time).abs().idxmin()
        close_count = ((candidates - response_times.loc[nearest_index]).abs() <= pd.Timedelta(minutes=5)).sum()
        if close_count > 1:
            match_warning = "Multiple same-date form responses are within 5 minutes of the selected match."
    elif same_date.any():
        candidates = response_times[same_date]
        nearest_index = (candidates - session_time).abs().idxmin()
        match_warning = "No same-date form response occurs after session start; using closest same-date response."
    else:
        nearest_index = (response_times - session_time).abs().idxmin()
        match_warning = "No same-date form response found; using closest timestamp across all dates."

    selected = raw.loc[[nearest_index]].copy()
    selected["MatchedResponseTimestamp"] = response_times.loc[nearest_index]
    selected["MinutesFromSessionStart"] = (response_times.loc[nearest_index] - session_time).total_seconds() / 60
    selected["MatchWarning"] = match_warning
    return selected


def _split_emotions(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []

    text = str(value).strip()
    found = [emotion for emotion in EMOTION_LABELS if re.search(rf"\b{re.escape(emotion)}\b", text, re.IGNORECASE)]
    if found:
        other_text = re.sub("|".join(re.escape(emotion) for emotion in found), "", text, flags=re.IGNORECASE)
        other_text = re.sub(r"[,;/\n]+", " ", other_text).strip(" :-")
        return found + ([f"Other: {other_text}"] if other_text else [])

    parts = [part.strip() for part in re.split(r"[,;/\n]+", text) if part.strip()]
    return parts or [text]


def _video_number_from_column(column: str) -> int | None:
    match = re.search(r"\bvideo\s*0*(\d+)\b", column, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _google_forms_repeat_number(column: str) -> int | None:
    match = re.search(r"\.(\d+)$", column)
    return int(match.group(1)) + 1 if match else None


def _question_kind(column: str) -> str | None:
    normalized = re.sub(r"\s+", " ", column.lower())
    if "intens" in normalized:
        return "intensity"
    if "how much" in normalized or "portion" in normalized or "how long" in normalized:
        return "duration"
    if "other" in normalized:
        return "other"
    if "emotion" in normalized or "feel" in normalized:
        return "emotion"
    return None


def _read_emotion_flow_file(flow_path: Path, phase: str) -> pd.DataFrame:
    rows = []
    if not flow_path.exists():
        return pd.DataFrame()

    pattern = re.compile(r"^\s*(\d+)\.\s*([A-Za-z]+)\s*-\s*([A-Za-z]+)\s*-\s*(.+?)\s*$")
    for line in flow_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        order, line_phase, emotion, recorded_at = match.groups()
        if line_phase.lower() != phase.lower():
            continue
        rows.append({
            "SessionID": flow_path.parent.name,
            "VideoNumber": int(order),
            "SourcePhase": line_phase,
            "StimulusEmotion": emotion,
            "FlowRecordedAt": recorded_at,
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_video_emotion_flow(project_root_str: str, session_id: str) -> pd.DataFrame:
    project_root = Path(project_root_str)
    flow_path = project_root / "VideoRecordings" / session_id / "emotion_flow.txt"
    return _read_emotion_flow_file(flow_path, "Video")


def discover_form_response_files(form_dir: Path) -> dict[str, Path]:
    if not form_dir.exists():
        return {}
    return {csv_file.stem: csv_file for csv_file in sorted(form_dir.glob("*.csv"))}


@st.cache_data(show_spinner=False)
def load_form_responses(form_file_str: str, session_id: str) -> pd.DataFrame:
    form_path = Path(form_file_str)
    if not form_path.exists():
        return pd.DataFrame()

    raw = pd.read_csv(form_path)
    if raw.empty:
        return pd.DataFrame()

    raw = _select_response_rows(raw, session_id)

    column_map: dict[int, dict[str, str]] = {}
    fallback_counts: dict[str, int] = {}
    for column in raw.columns:
        video_number = _video_number_from_column(column)
        kind = _question_kind(column)
        if video_number is None or kind is None:
            if kind is None:
                continue
            video_number = _google_forms_repeat_number(column)
            if video_number is None:
                fallback_counts[kind] = fallback_counts.get(kind, 0) + 1
                video_number = fallback_counts[kind]
        column_map.setdefault(video_number, {})[kind] = column

    rows = []
    timestamp_col = _timestamp_column(raw.columns)
    for response_index, response in raw.iterrows():
        response_timestamp = response.get("MatchedResponseTimestamp", response.get(timestamp_col, ""))
        minutes_from_session = response.get("MinutesFromSessionStart", np.nan)
        match_warning = response.get("MatchWarning", "")
        for video_number in sorted(column_map):
            columns = column_map[video_number]
            emotion_raw = response.get(columns.get("emotion"), np.nan)
            duration_raw = response.get(columns.get("duration"), np.nan)
            intensity_raw = response.get(columns.get("intensity"), np.nan)
            other_raw = response.get(columns.get("other"), np.nan)

            reported_emotions = _split_emotions(emotion_raw)
            if not pd.isna(other_raw) and str(other_raw).strip():
                reported_emotions.append(f"Other: {str(other_raw).strip()}")

            rows.append({
                "SessionID": session_id,
                "ResponseRow": response_index + 1,
                "ResponseTimestamp": "" if pd.isna(response_timestamp) else str(response_timestamp),
                "MinutesFromSessionStart": minutes_from_session,
                "MatchWarning": "" if pd.isna(match_warning) else str(match_warning),
                "VideoNumber": video_number,
                "ReportedEmotionRaw": "" if pd.isna(emotion_raw) else str(emotion_raw),
                "ReportedEmotionList": reported_emotions,
                "ReportedEmotion": ", ".join(reported_emotions),
                "DurationRaw": "" if pd.isna(duration_raw) else str(duration_raw),
                "DurationScore": _parse_score(duration_raw),
                "IntensityRaw": "" if pd.isna(intensity_raw) else str(intensity_raw),
                "IntensityScore": _parse_score(intensity_raw),
            })

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def load_and_process(project_root_str: str, session_id: str) -> pd.DataFrame:
    project_root = Path(project_root_str)
    video_dir = project_root / "VideoRecordings" / session_id
    acting_dir = project_root / "ActingRecordings" / session_id
    if not acting_dir.exists():
        acting_dir = project_root / "ActingRecordings"

    frames = []
    if video_dir.exists():
        for csv_file in sorted(video_dir.glob("*/weights.csv")):
            df = pd.read_csv(csv_file)
            df["SourceEmotion"] = csv_file.parent.name
            df["SourcePhase"] = "Video"
            frames.append(df)
    if acting_dir.exists():
        for csv_file in sorted(acting_dir.glob("*/acting.csv")):
            df = pd.read_csv(csv_file)
            df["SourceEmotion"] = csv_file.parent.name
            df["SourcePhase"] = "Acting"
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "Timestamp" not in combined.columns:
        return combined

    results = []
    for (_, _), group in combined.groupby(["SourceEmotion", "SourcePhase"]):
        group = group.sort_values("Timestamp").copy()
        group["ElapsedSeconds"] = group["Timestamp"] - group["Timestamp"].min()
        dt = group["Timestamp"].diff()
        for p in SIGNAL_PREFIXES:
            x, y, z = f"{p}X", f"{p}Y", f"{p}Z"
            if not {x, y, z}.issubset(group.columns):
                continue
            dist = np.sqrt(group[x].diff() ** 2 + group[y].diff() ** 2 + group[z].diff() ** 2)
            group[f"{p}Speed"] = dist / dt.replace(0, np.nan)
        results.append(group)
    return pd.concat(results, ignore_index=True) if results else combined


def metric_card(label: str, value: str) -> None:
    st.metric(label, value)


def first_existing(columns: list[str], options: list[str]) -> str | None:
    return next((option for option in options if option in columns), None)


def available_phases(dff: pd.DataFrame) -> list[str]:
    return [phase for phase in PHASE_ORDER if phase in set(dff["SourcePhase"])]


def plot_line_by_phase(df: pd.DataFrame, y_col: str, title: str, y_label: str, height: int = 380) -> None:
    fig = px.line(
        df,
        x="ElapsedSeconds",
        y=y_col,
        color="SourceEmotion",
        facet_col="SourcePhase",
        category_orders={"SourcePhase": PHASE_ORDER},
        title=title,
        labels={"ElapsedSeconds": "Time from start (s)", y_col: y_label},
        height=height,
    )
    fig.update_traces(opacity=0.75)
    fig.update_xaxes(matches=None)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Showing all {len(df):,} rows in this chart.")


def wrap_degrees(series: pd.Series) -> pd.Series:
    return ((series + 180) % 360) - 180


def position_and_speed_section(dff: pd.DataFrame, prefix: str, label: str) -> None:
    axes = [f"{prefix}X", f"{prefix}Y", f"{prefix}Z"]
    speed_col = f"{prefix}Speed"

    if not any(c in dff.columns for c in axes):
        st.info(f"No position data for {label}.")
        return

    available_axes = [axis for axis in axes if axis in dff.columns]
    axis = st.selectbox("Position axis", available_axes, key=f"{prefix}_axis")
    plot_line_by_phase(dff, axis, f"{axis} over Time", axis, height=360)

    if speed_col in dff.columns:
        speed_df = dff[dff[speed_col].notna()]
        if speed_df.empty:
            st.info(f"No speed values available for {label}.")
        else:
            fig = px.box(
                speed_df,
                x="SourceEmotion",
                y=speed_col,
                color="SourceEmotion",
                facet_col="SourcePhase",
                category_orders={"SourcePhase": PHASE_ORDER},
                points=False,
                title=f"{label} Speed by Emotion",
                labels={speed_col: "Speed (units/s)", "SourceEmotion": "Emotion"},
            )
            fig.update_xaxes(matches=None)
            st.plotly_chart(fig, width="stretch")


def head_rotation_section(dff: pd.DataFrame) -> None:
    available_rotation_cols = [col for col in HEAD_ROTATION_COLS if col in dff.columns]
    if not available_rotation_cols:
        st.info("No head rotation columns found. New recordings will include HeadPitch, HeadYaw, and HeadRoll.")
        return

    rotation_col = st.selectbox("Rotation axis", available_rotation_cols)
    rotation_df = dff.copy()
    wrapped_col = f"{rotation_col}Wrapped"
    rotation_df[wrapped_col] = wrap_degrees(rotation_df[rotation_col])
    plot_line_by_phase(rotation_df, wrapped_col, f"{rotation_col} over Time", "Degrees (-180 to 180)", height=360)


def eye_gaze_section(dff: pd.DataFrame) -> None:
    if "EyeGazeTracked" not in dff.columns:
        st.info("No eye gaze columns found. New Quest Pro recordings can include eye gaze when eye tracking is available.")
        return

    gaze_df = dff[dff["EyeGazeTracked"] == 1]
    if gaze_df.empty:
        st.info("Eye gaze columns are present, but no tracked gaze samples are available for this selection.")
        return

    st.caption(f"{len(gaze_df):,} tracked eye-gaze frames out of {len(dff):,} filtered")
    available_direction_cols = [col for col in ["EyeGazeDirX", "EyeGazeDirY", "EyeGazeDirZ"] if col in gaze_df.columns]
    if available_direction_cols:
        direction_col = st.selectbox("Gaze direction axis", available_direction_cols)
        plot_line_by_phase(gaze_df, direction_col, f"{direction_col} over Time", direction_col, height=360)

    available_rotation_cols = [col for col in ["EyeGazePitch", "EyeGazeYaw", "EyeGazeRoll"] if col in gaze_df.columns]
    if available_rotation_cols:
        rotation_col = st.selectbox("Gaze rotation axis", available_rotation_cols)
        rotation_df = gaze_df.copy()
        wrapped_col = f"{rotation_col}Wrapped"
        rotation_df[wrapped_col] = wrap_degrees(rotation_df[rotation_col])
        plot_line_by_phase(rotation_df, wrapped_col, f"{rotation_col} over Time", "Degrees (-180 to 180)", height=360)


def form_responses_dashboard(form_df: pd.DataFrame, flow_df: pd.DataFrame, form_file: Path, form_session_id: str) -> None:
    st.caption(f"Form session `{form_session_id}`")
    st.caption(f"Forms CSV path: `{form_file}`")

    if flow_df.empty:
        st.warning("No video `emotion_flow.txt` found for this form session. The form can still be shown, but video numbers cannot be mapped to randomized stimulus emotions.")
    else:
        st.subheader("Randomized Video Order")
        st.dataframe(
            flow_df[["VideoNumber", "StimulusEmotion", "FlowRecordedAt"]],
            width="stretch",
            hide_index=True,
        )

    if form_df.empty:
        st.info("No form response data found. Check that `Forms.csv` exists and has a parseable `Timestamp` column.")
        return

    if "ResponseTimestamp" in form_df.columns and form_df["ResponseTimestamp"].notna().any():
        response_row = int(form_df["ResponseRow"].iloc[0])
        timestamp = form_df["ResponseTimestamp"].dropna().iloc[0]
        minutes = form_df["MinutesFromSessionStart"].dropna()
        if not minutes.empty:
            st.caption(f"Matched Forms.csv row: `{response_row}` | timestamp: `{timestamp}` ({minutes.iloc[0]:+.1f} minutes from session start)")
        else:
            st.caption(f"Matched Forms.csv row: `{response_row}` | timestamp: `{timestamp}`")

    if "MatchWarning" in form_df.columns:
        warnings = [warning for warning in form_df["MatchWarning"].dropna().unique() if str(warning).strip()]
        for warning in warnings:
            st.warning(warning)

    if form_df["ResponseRow"].nunique() > 1:
        st.warning("Multiple form response rows are being shown. If the CSV has a Timestamp column, the loader should normally select the closest response to the form session ID.")

    trial_df = form_df.copy()
    if not flow_df.empty:
        trial_df = trial_df.merge(
            flow_df[["VideoNumber", "StimulusEmotion", "FlowRecordedAt"]],
            on="VideoNumber",
            how="left",
        )
    else:
        trial_df["StimulusEmotion"] = ""
        trial_df["FlowRecordedAt"] = ""

    trial_df["MissingAnswer"] = (
        trial_df["ReportedEmotion"].eq("")
        | trial_df["DurationScore"].isna()
        | trial_df["IntensityScore"].isna()
    )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        metric_card("Form videos", str(trial_df["VideoNumber"].nunique()))
    with metric_cols[1]:
        metric_card("Missing answers", str(int(trial_df["MissingAnswer"].sum())))

    display_cols = [
        "ResponseTimestamp",
        "VideoNumber",
        "StimulusEmotion",
        "ReportedEmotion",
        "DurationRaw",
        "IntensityRaw",
        "MissingAnswer",
    ]
    st.subheader("Trial-Level Form Data")
    st.dataframe(
        trial_df[[col for col in display_cols if col in trial_df.columns]],
        width="stretch",
        hide_index=True,
    )

    missing_df = trial_df[trial_df["MissingAnswer"]]
    if not missing_df.empty:
        st.warning("Some form responses are missing emotion, duration, or intensity values.")
        st.dataframe(
            missing_df[["VideoNumber", "StimulusEmotion", "ReportedEmotion", "DurationRaw", "IntensityRaw"]],
            width="stretch",
            hide_index=True,
        )

    exploded = trial_df.explode("ReportedEmotionList").rename(columns={"ReportedEmotionList": "ReportedEmotionItem"})
    exploded = exploded[exploded["ReportedEmotionItem"].notna() & (exploded["ReportedEmotionItem"].astype(str).str.strip() != "")]

    if not exploded.empty and exploded["StimulusEmotion"].ne("").any():
        st.subheader("Stimulus Emotion vs Reported Emotion")
        confusion = (
            exploded.groupby(["StimulusEmotion", "ReportedEmotionItem"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.density_heatmap(
            confusion,
            x="ReportedEmotionItem",
            y="StimulusEmotion",
            z="Count",
            histfunc="sum",
            text_auto=True,
            title="Self-Reported Emotion Selections by Stimulus Emotion",
            labels={"ReportedEmotionItem": "Reported Emotion", "StimulusEmotion": "Stimulus Emotion"},
            height=420,
        )
        st.plotly_chart(fig, width="stretch")


def set_source_mode(mode: str) -> None:
    st.session_state["source_mode"] = mode


# Page
st.set_page_config(page_title="VR Emotion Study", layout="wide")
st.title("VR Emotion Study - Data Visualization")

# Sidebar
with st.sidebar:
    st.header("Data source")
    st.session_state.setdefault("source_mode", "Recordings")
    recordings_tab, forms_tab = st.tabs(["Recordings", "Forms"])

    with recordings_tab:
        st.button("Show recording visualization", on_click=set_source_mode, args=("Recordings",), width="stretch")
        project_root_str = st.text_input(
            "Project root path",
            str(_DEFAULT_ROOT),
            on_change=set_source_mode,
            args=("Recordings",),
        )
        project_root = Path(project_root_str)

        sessions = _discover_sessions(project_root)
        if sessions:
            session_id = st.selectbox(
                "Session",
                sessions,
                index=len(sessions) - 1,
                on_change=set_source_mode,
                args=("Recordings",),
            )
        else:
            session_id = st.text_input("Session ID (manual)", "", on_change=set_source_mode, args=("Recordings",))

    default_form_dir = _DEFAULT_FORM_DIR
    with forms_tab:
        st.button("Show form visualization", on_click=set_source_mode, args=("Forms",), width="stretch")
        default_form_file = default_form_dir / "Forms.csv"
        form_file_str = st.text_input(
            "Form responses file",
            str(default_form_file),
            on_change=set_source_mode,
            args=("Forms",),
        )
        form_file = Path(form_file_str)
        form_sessions = sorted(sessions)
        if form_sessions:
            form_session_id = st.selectbox(
                "Form session",
                form_sessions,
                index=len(form_sessions) - 1,
                on_change=set_source_mode,
                args=("Forms",),
            )
        else:
            form_session_id = st.text_input("Form session", "", on_change=set_source_mode, args=("Forms",))

        if not form_session_id:
            st.info("Select a form session.")
            st.stop()

        st.caption(f"Expected file: `{form_file.name}`")
        if form_file.exists():
            st.success("Forms.csv found.")
        else:
            st.warning("Forms.csv not found.")

    source_mode = st.session_state["source_mode"]
    if source_mode == "Forms":
        form_df = load_form_responses(form_file_str, form_session_id)
        form_flow_df = load_video_emotion_flow(project_root_str, form_session_id)
    else:
        if not session_id:
            st.info("Select a session to load data.")
            st.stop()

        with st.spinner("Loading and preparing recording data..."):
            df = load_and_process(project_root_str, session_id)
        if df.empty:
            st.error("No data found. Check Session ID and data paths.")
            st.stop()

        all_emotions = sorted(df["SourceEmotion"].unique())
        st.header("Filters")
        sel_emotions = st.multiselect("Emotions", all_emotions, default=all_emotions)

        if not sel_emotions:
            st.info("Select at least one emotion.")
            st.stop()

        st.caption(f"{len(df):,} total frames | {len(all_emotions)} emotions")

if source_mode == "Forms":
    st.subheader("Form Response Visualization")
    form_responses_dashboard(form_df, form_flow_df, form_file, form_session_id)
    st.stop()

dff = df[df["SourceEmotion"].isin(sel_emotions)]

if dff.empty:
    st.warning("No data matches the current filters.")
    st.stop()

st.caption(f"Session `{session_id}` | {len(dff):,} filtered frames")

metric_cols = st.columns(4)
with metric_cols[0]:
    metric_card("Filtered frames", f"{len(dff):,}")
with metric_cols[1]:
    metric_card("Emotions", str(dff["SourceEmotion"].nunique()))
with metric_cols[2]:
    metric_card("Data sources", ", ".join(available_phases(dff)))
with metric_cols[3]:
    time_col = first_existing(list(dff.columns), ["Timestamp", "RealtimeSeconds"])
    if time_col:
        duration = dff.groupby(["SourceEmotion", "SourcePhase"])[time_col].agg(lambda s: s.max() - s.min()).sum()
        metric_card("Recorded time", f"{duration:,.1f}s")
    else:
        metric_card("Recorded time", "n/a")

# AU column detection
_non_au = (
    {"Timestamp", "ElapsedSeconds", "RealtimeSeconds", "Emotion", "SourceEmotion", "SourcePhase",
     "SourceLabel", "SourceFile", "FaceValid"}
    | {f"{p}Tracked" for p in SIGNAL_PREFIXES}
    | {f"{p}{ax}" for p in SIGNAL_PREFIXES for ax in ["X", "Y", "Z"]}
    | {f"{p}Speed" for p in SIGNAL_PREFIXES}
    | {f"{p}Distance" for p in SIGNAL_PREFIXES}
    | set(HEAD_ROTATION_COLS)
    | set(HEAD_QUATERNION_COLS)
    | set(EYE_GAZE_COLS)
)
AU_COLS = [c for c in df.columns if c not in _non_au]

view = st.radio(
    "View",
    ["Overview", "Face", "Head", "Gaze", "Hands", "Controllers", "Quality"],
    horizontal=True,
)

if view == "Overview":
    st.subheader("Emotion and Phase Coverage")
    counts = (
        dff.groupby(["SourceEmotion", "SourcePhase"])
        .size()
        .reset_index(name="Frames")
    )
    fig = px.bar(
        counts,
        x="SourceEmotion",
        y="Frames",
        color="SourcePhase",
        category_orders={"SourcePhase": PHASE_ORDER},
        barmode="group",
        title="Frames by Emotion and Phase",
        height=360,
    )
    st.plotly_chart(fig, width="stretch")

    if AU_COLS:
        au_summary = (
            dff.groupby(["SourcePhase", "SourceEmotion"])[AU_COLS]
            .mean(numeric_only=True)
            .reset_index()
            .melt(
                id_vars=["SourcePhase", "SourceEmotion"],
                var_name="Action Unit",
                value_name="Mean Weight",
            )
        )
        top_aus = (
            au_summary.groupby("Action Unit")["Mean Weight"]
            .mean()
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        st.subheader("Top Mean Facial Action Unit Weights")
        phases = available_phases(dff)
        phase_cols = st.columns(len(phases) or 1)
        for col, phase in zip(phase_cols, phases):
            phase_summary = au_summary[
                (au_summary["SourcePhase"] == phase)
                & (au_summary["Action Unit"].isin(top_aus))
            ]
            with col:
                fig = px.imshow(
                    phase_summary
                    .pivot(index="SourceEmotion", columns="Action Unit", values="Mean Weight")
                    .fillna(0),
                    aspect="auto",
                    title=phase,
                    labels=dict(x="Action Unit", y="Emotion", color="Mean Weight"),
                    height=420,
                )
                st.plotly_chart(fig, width="stretch")

elif view == "Face":
    st.subheader("Facial Action Units")
    if not AU_COLS:
        st.info("No AU columns detected.")
    else:
        face_df = dff[dff["FaceValid"] == 1] if "FaceValid" in dff.columns else dff
        st.caption(f"{len(face_df):,} valid-face frames out of {len(dff):,} filtered")

        au = st.selectbox("Action Unit", AU_COLS)
        plot_line_by_phase(face_df, au, f"{au} over Time", "Weight", height=420)

        st.subheader(f"Mean {au} per Emotion")
        mean_df = face_df.groupby(["SourceEmotion", "SourcePhase"])[au].mean().reset_index()
        fig2 = px.bar(
            mean_df,
            x="SourceEmotion",
            y=au,
            color="SourceEmotion",
            facet_col="SourcePhase",
            category_orders={"SourcePhase": PHASE_ORDER},
            labels={au: "Mean Weight", "SourceEmotion": "Emotion"},
            height=350,
        )
        fig2.update_xaxes(matches=None)
        st.plotly_chart(fig2, width="stretch")

elif view == "Head":
    st.subheader("Head Position & Speed")
    position_and_speed_section(dff, "Head", "Head")
    st.subheader("Head Rotation")
    head_rotation_section(dff)

elif view == "Gaze":
    st.subheader("Eye Gaze")
    eye_gaze_section(dff)

elif view == "Hands":
    st.subheader("Hand Position & Speed")
    hand_choice = st.radio("Show", ["Left Hand", "Right Hand", "Both"], horizontal=True)
    if hand_choice in ("Left Hand", "Both"):
        st.markdown("#### Left Hand")
        position_and_speed_section(dff, "LeftHand", "Left Hand")
    if hand_choice in ("Right Hand", "Both"):
        st.markdown("#### Right Hand")
        position_and_speed_section(dff, "RightHand", "Right Hand")

elif view == "Controllers":
    st.subheader("Controller Speed")
    ctrl_choice = st.radio("Show", ["Left", "Right", "Both"], horizontal=True)
    if ctrl_choice in ("Left", "Both"):
        st.markdown("#### Left Controller")
        position_and_speed_section(dff, "LeftController", "Left Controller")
    if ctrl_choice in ("Right", "Both"):
        st.markdown("#### Right Controller")
        position_and_speed_section(dff, "RightController", "Right Controller")

elif view == "Quality":
    st.subheader("Recording Quality - Tracking Rates")
    tracked_cols = [c for c in [f"{p}Tracked" for p in SIGNAL_PREFIXES] if c in dff.columns]
    tracked_cols += [c for c in ["EyeGazeTracked"] if c in dff.columns]
    agg_cols = tracked_cols + (["FaceValid"] if "FaceValid" in dff.columns else [])

    if not agg_cols:
        st.info("No tracking-quality columns found in the filtered data.")
    else:
        quality_df = (
            dff.groupby(["SourceEmotion", "SourcePhase"])[agg_cols]
            .mean()
            .multiply(100)
            .round(1)
            .reset_index()
        )
        rename_map = {c: c.replace("Tracked", " %") for c in tracked_cols}
        if "FaceValid" in agg_cols:
            rename_map["FaceValid"] = "FaceValid %"
        quality_df = quality_df.rename(columns=rename_map)

        st.dataframe(quality_df, width="stretch")

        rate_cols = list(rename_map.values())
        melted = quality_df.melt(
            id_vars=["SourceEmotion", "SourcePhase"],
            value_vars=rate_cols,
            var_name="Signal",
            value_name="Tracking Rate (%)",
        )
        fig = px.bar(
            melted,
            x="SourceEmotion",
            y="Tracking Rate (%)",
            color="Signal",
            facet_col="SourcePhase",
            category_orders={"SourcePhase": PHASE_ORDER},
            barmode="group",
            title="Tracking Rate (%) by Emotion, Phase & Signal",
            height=450,
        )
        st.plotly_chart(fig, width="stretch")
