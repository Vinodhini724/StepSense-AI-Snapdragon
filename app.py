from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core.io import load_attempt, load_procedure
from core.sequence_engine import Observation, evaluate_attempt, feedback_messages


ROOT = Path(__file__).resolve().parent
PROCEDURE_PATH = ROOT / "data" / "procedures" / "lab_safety.json"
DEMO_DIR = ROOT / "data" / "demo_attempts"

st.set_page_config(page_title="StepSense AI", page_icon="✅", layout="wide")
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.7rem; padding-bottom: 2rem;}
      .hero {padding: 1.3rem 1.5rem; border-radius: 18px; background: linear-gradient(115deg,#0f2747,#194f73); color:white; margin-bottom:1rem;}
      .hero h1 {margin:0; font-size:2.25rem;}
      .hero p {margin:.4rem 0 0; color:#d7edf7; font-size:1.05rem;}
      .small-note {color:#667085; font-size:.88rem;}
    </style>
    <div class="hero"><h1>StepSense AI</h1><p>On-device practical skill assessment for Snapdragon PCs</p></div>
    """,
    unsafe_allow_html=True,
)

procedure, steps = load_procedure(PROCEDURE_PATH)
name_by_id = {step.id: step.name for step in steps}

with st.sidebar:
    st.header("Procedure")
    st.write(procedure["name"])
    for index, step in enumerate(steps, start=1):
        st.write(f"{index}. {step.name}")
    st.caption("All assessment data remains local in this prototype.")

overview, assess, technical = st.tabs(["Overview", "Run assessment", "Technical details"])

with overview:
    st.subheader("Evaluate the process, not only the result")
    st.write(
        "StepSense compares detected actions with an approved procedure graph and explains "
        "missed, repeated, delayed and out-of-order steps. The current MVP demonstrates a "
        "five-step laboratory safety preparation routine."
    )
    cols = st.columns(4)
    for col, value, label in zip(cols, ["5", "4", "Local", "Explainable"],
                                 ["Procedure steps", "Demo scenarios", "Processing design", "Feedback"]):
        col.metric(label, value)
    st.info("Use Demo mode to test the complete validation workflow without downloading an AI model.")

with assess:
    mode = st.radio("Input mode", ["Demo attempt", "Manual detected sequence", "Video AI experiment"], horizontal=True)
    observations = None
    attempt_name = ""

    if mode == "Demo attempt":
        files = sorted(DEMO_DIR.glob("*.json"))
        labels = {}
        for path in files:
            payload, _ = load_attempt(path)
            labels[payload["name"]] = path
        selected = st.selectbox("Choose an attempt", list(labels))
        attempt, observations = load_attempt(labels[selected])
        attempt_name = attempt["name"]

    elif mode == "Manual detected sequence":
        st.caption("Simulate the output of an action-recognition model by arranging detected steps.")
        selected_steps = st.multiselect(
            "Detected actions in observed order",
            options=[step.id for step in steps],
            default=[step.id for step in steps],
            format_func=lambda value: name_by_id[value],
        )
        observations = [
            Observation(step_id=sid, start_seconds=i * 8, end_seconds=(i + 1) * 8, confidence=0.90)
            for i, sid in enumerate(selected_steps)
        ]
        attempt_name = "Manual sequence"

    else:
        uploaded = st.file_uploader("Upload a short MP4 assessment video", type=["mp4", "mov", "avi"])
        st.warning(
            "Experimental adapter: first use downloads an open-source CLIP model. For the final "
            "submission, replace it with a fine-tuned ONNX/QNN action model and report Snapdragon latency."
        )
        if uploaded and st.button("Analyse video locally", type="primary"):
            from core.video_ai import analyse_video_zero_shot
            try:
                with st.spinner("Downloading/loading CLIP and classifying sampled frames..."):
                    observations = analyse_video_zero_shot(uploaded.getvalue(), steps)
                st.session_state["video_observations"] = observations
            except Exception as exc:
                st.error("The video model could not be loaded or the video could not be processed.")
                st.code(str(exc), language=None)
        observations = observations or st.session_state.get("video_observations")
        attempt_name = uploaded.name if uploaded else "Video assessment"

    if observations is not None and st.button("Generate assessment report", type="primary", use_container_width=True):
        st.session_state["result"] = evaluate_attempt(steps, observations)
        st.session_state["attempt_name"] = attempt_name

    result = st.session_state.get("result")
    if result:
        st.divider()
        st.subheader(st.session_state.get("attempt_name", "Assessment report"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Overall score", f'{result["overall_score"]}%')
        c2.metric("Completion", f'{result["completion_score"]}%')
        c3.metric("Sequence accuracy", f'{result["sequence_score"]}%')
        c4.metric("Status", result["status"])

        if result["status"] == "Completed":
            st.success(feedback_messages(result)[0])
        else:
            for message in feedback_messages(result):
                st.warning(message)

        timeline = pd.DataFrame(result["timeline"])
        if not timeline.empty:
            timeline["confidence"] = (timeline["confidence"] * 100).round(1).astype(str) + "%"
            st.dataframe(
                timeline[["order", "step_name", "start_seconds", "end_seconds", "duration_seconds", "confidence"]],
                use_container_width=True,
                hide_index=True,
            )
        st.download_button(
            "Download JSON report",
            data=json.dumps(result, indent=2),
            file_name="stepsense_assessment_report.json",
            mime="application/json",
        )

with technical:
    st.subheader("Prototype architecture")
    st.markdown(
        """
        1. **Video input** samples the learner's attempt.
        2. **Vision adapter** converts frames into action observations.
        3. **Sequence engine** compares observations with the approved process graph.
        4. **Deviation logic** detects missing, repeated, delayed and out-of-order steps.
        5. **Report layer** provides scores, a timeline and trainer-review flags.
        """
    )
    st.code("Camera → Action recognition → Procedure graph → Deviation engine → Report", language=None)
    st.caption("The deterministic sequence engine is model-independent, allowing the vision model to be replaced during Snapdragon optimisation.")
