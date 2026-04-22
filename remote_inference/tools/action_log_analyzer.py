"""Streamlit dashboard for analyzing remote_inference per-action CSV logs.

Usage:
    streamlit run vt_src/remote_inference/tools/action_log_analyzer.py

In the sidebar: paste the path to the CSV (default: ./remote_inference_actions.csv).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="Remote Inference Action Analyzer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar — CSV selection
# ---------------------------------------------------------------------------
st.sidebar.header("Input")
default_csv = "remote_inference_actions.csv"
csv_path_str = st.sidebar.text_input("CSV path", value=default_csv)
csv_path = Path(csv_path_str).expanduser()

if not csv_path.exists():
    st.error(f"CSV not found at: {csv_path.resolve()}")
    st.stop()

df = pd.read_csv(csv_path)
joints = [c for c in df.columns if c.endswith(".pos") or c.startswith("m")]
joints = [j for j in joints if j not in {"t_ms", "chunk_id", "action_idx", "chunk_fire_ms", "age_ms"}]

# ---------------------------------------------------------------------------
# Summary strip
# ---------------------------------------------------------------------------
st.title("Remote Inference Action Log")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Duration", f"{df['t_ms'].iloc[-1] / 1000:.1f} s")
c3.metric("Chunks", int(df["chunk_id"].max()))
dt_ms = df["t_ms"].diff().median()
c4.metric("Pop rate", f"{1000 / dt_ms:.1f} Hz" if dt_ms else "—")
c5.metric("Median age", f"{df['age_ms'].median():.0f} ms")

st.sidebar.header("Filters")
t_max = float(df["t_ms"].iloc[-1])
t_range = st.sidebar.slider("Time range (ms)", 0.0, t_max, (0.0, t_max), step=100.0)
df = df[(df["t_ms"] >= t_range[0]) & (df["t_ms"] <= t_range[1])].reset_index(drop=True)

selected_joints = st.sidebar.multiselect("Joints to display", joints, default=joints)

show_chunk_lines = st.sidebar.checkbox("Show chunk boundaries (first action popped)", value=True)
show_fire_lines = st.sidebar.checkbox("Show inference fire times", value=True)
show_boundary_markers = st.sidebar.checkbox("Highlight transition points", value=True)

st.sidebar.header("Display")
window_s = st.sidebar.number_input(
    "Initial x-window (seconds)",
    min_value=1.0,
    max_value=120.0,
    value=10.0,
    step=1.0,
    help="Time-series plots start zoomed to this width; use the range slider or scroll to pan.",
)

# ---------------------------------------------------------------------------
# Helper: identify chunk-boundary rows (first row of each new chunk)
# ---------------------------------------------------------------------------
df["_new_chunk"] = df["chunk_id"].diff() > 0
boundary_times = df.loc[df["_new_chunk"], "t_ms"].tolist()
boundary_chunk_ids = df.loc[df["_new_chunk"], "chunk_id"].tolist()

# Unique fire times: each chunk fired exactly once, at chunk_fire_ms.
fire_df = df.groupby("chunk_id", as_index=False)["chunk_fire_ms"].first()
fire_times = fire_df["chunk_fire_ms"].tolist()
fire_chunk_ids = fire_df["chunk_id"].tolist()

# Initial x-range so plots start zoomed in (user pans via the range slider).
x_initial_start = df["t_ms"].iloc[0]
x_initial_end = min(x_initial_start + window_s * 1000, df["t_ms"].iloc[-1])
x_full_start = df["t_ms"].iloc[0]
x_full_end = df["t_ms"].iloc[-1]


def _add_fire_and_boundary_lines(fig):
    """Attach vertical lines for chunk boundaries (grey dotted) and fire times (orange dashed)."""
    if show_chunk_lines:
        for t in boundary_times:
            fig.add_vline(x=t, line=dict(color="rgba(120,120,120,0.3)", dash="dot", width=1))
    if show_fire_lines:
        for t in fire_times:
            fig.add_vline(x=t, line=dict(color="rgba(255,140,0,0.55)", dash="dash", width=1))


def _apply_slider_layout(fig, yaxis_title: str, height: int):
    fig.update_layout(
        height=height,
        xaxis_title="t (ms)",
        yaxis_title=yaxis_title,
        hovermode="x unified",
        margin=dict(t=30, l=40, r=40, b=40),
        xaxis=dict(
            range=[x_initial_start, x_initial_end],
            rangeslider=dict(visible=True, thickness=0.08, range=[x_full_start, x_full_end]),
            type="linear",
        ),
    )

# ---------------------------------------------------------------------------
# Panel 1: Time series per joint + chunk boundaries
# ---------------------------------------------------------------------------
st.header("1. Joint trajectories over time")
st.caption(
    "Grey dotted = first-action-popped (chunk boundary). "
    "Orange dashed = inference fire time (obs captured). "
    "Drag the range slider at the bottom to scroll; drag edges to zoom."
)
fig = go.Figure()
for j in selected_joints:
    fig.add_trace(go.Scatter(x=df["t_ms"], y=df[j], mode="lines", name=j, line=dict(width=1)))

_add_fire_and_boundary_lines(fig)

if show_boundary_markers:
    for j in selected_joints:
        bdy_df = df[df["_new_chunk"]]
        fig.add_trace(
            go.Scatter(
                x=bdy_df["t_ms"],
                y=bdy_df[j],
                mode="markers",
                marker=dict(size=6, symbol="circle-open"),
                name=f"{j} (boundary)",
                showlegend=False,
                opacity=0.6,
            )
        )

_apply_slider_layout(fig, yaxis_title="position (normalized)", height=500)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Panel 2: age_ms at execution (staleness) per chunk
# ---------------------------------------------------------------------------
st.header("2. Plan staleness at execution (age_ms)")
fig_age = go.Figure()
fig_age.add_trace(
    go.Scatter(
        x=df["t_ms"],
        y=df["age_ms"],
        mode="lines",
        name="age_ms",
        line=dict(color="crimson", width=1),
    )
)
_add_fire_and_boundary_lines(fig_age)
_apply_slider_layout(fig_age, yaxis_title="age (ms) — plan → execution lag", height=320)
st.plotly_chart(fig_age, use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Age min", f"{df['age_ms'].min():.0f} ms")
col2.metric("Age median", f"{df['age_ms'].median():.0f} ms")
col3.metric("Age max", f"{df['age_ms'].max():.0f} ms")

# ---------------------------------------------------------------------------
# Panel 3: Transition deltas per joint (last of N vs first of N+1)
# ---------------------------------------------------------------------------
st.header("3. Transition deltas per joint")

# Compute deltas: for each chunk N→N+1, take last row of N and first row of N+1
transitions = []
unique_chunks = sorted(df["chunk_id"].unique())
for i in range(len(unique_chunks) - 1):
    cid_a, cid_b = unique_chunks[i], unique_chunks[i + 1]
    last_a = df[df["chunk_id"] == cid_a].iloc[-1]
    first_b = df[df["chunk_id"] == cid_b].iloc[0]
    row = {"chunk_from": cid_a, "chunk_to": cid_b, "t_ms": first_b["t_ms"]}
    for j in joints:
        row[j] = first_b[j] - last_a[j]
    transitions.append(row)
trans_df = pd.DataFrame(transitions)

if len(trans_df) > 0:
    fig_delta = make_subplots(
        rows=len(selected_joints),
        cols=1,
        shared_xaxes=True,
        subplot_titles=selected_joints,
        vertical_spacing=0.02,
    )
    for idx, j in enumerate(selected_joints, start=1):
        fig_delta.add_trace(
            go.Bar(
                x=[f"{a}→{b}" for a, b in zip(trans_df["chunk_from"], trans_df["chunk_to"])],
                y=trans_df[j],
                name=j,
                showlegend=False,
            ),
            row=idx,
            col=1,
        )
    fig_delta.update_layout(
        height=180 * len(selected_joints),
        margin=dict(t=30, l=40, r=40, b=40),
    )
    st.plotly_chart(fig_delta, use_container_width=True)

    st.subheader("Transition summary (absolute delta per joint)")
    summary = pd.DataFrame(
        {
            "joint": selected_joints,
            "mean_abs_delta": [trans_df[j].abs().mean() for j in selected_joints],
            "p95_abs_delta": [trans_df[j].abs().quantile(0.95) for j in selected_joints],
            "max_abs_delta": [trans_df[j].abs().max() for j in selected_joints],
        }
    ).sort_values("max_abs_delta", ascending=False)
    st.dataframe(summary.style.format({"mean_abs_delta": "{:.2f}", "p95_abs_delta": "{:.2f}", "max_abs_delta": "{:.2f}"}), use_container_width=True)

# ---------------------------------------------------------------------------
# Panel 4: Histogram of transition deltas per joint
# ---------------------------------------------------------------------------
st.header("4. Transition delta distribution")
if len(trans_df) > 0:
    fig_hist = go.Figure()
    for j in selected_joints:
        fig_hist.add_trace(
            go.Histogram(
                x=trans_df[j],
                name=j,
                opacity=0.55,
                nbinsx=30,
            )
        )
    fig_hist.update_layout(
        barmode="overlay",
        height=400,
        xaxis_title="Δ position at chunk boundary",
        yaxis_title="count",
        margin=dict(t=30, l=40, r=40, b=40),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ---------------------------------------------------------------------------
# Raw data peek
# ---------------------------------------------------------------------------
with st.expander("Raw CSV preview"):
    st.dataframe(df.head(200), use_container_width=True)
