import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).parent.parent / "multimodal-vr-emotion-study-main"
SIGNAL_PREFIXES = ["Head", "LeftHand", "RightHand", "LeftController", "RightController"]
PHASE_ORDER = ["Video", "Acting"]


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


# Page
st.set_page_config(page_title="VR Emotion Study", layout="wide")
st.title("VR Emotion Study - Data Visualization")

# Sidebar
with st.sidebar:
    st.header("Data source")
    project_root_str = st.text_input("Project root path", str(_DEFAULT_ROOT))
    project_root = Path(project_root_str)

    sessions = _discover_sessions(project_root)
    if sessions:
        session_id = st.selectbox("Session", sessions, index=len(sessions) - 1)
    else:
        session_id = st.text_input("Session ID (manual)", "")

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
)
AU_COLS = [c for c in df.columns if c not in _non_au]

view = st.radio(
    "View",
    ["Overview", "Face", "Head", "Hands", "Controllers", "Quality"],
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
