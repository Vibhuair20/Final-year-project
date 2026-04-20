#!/usr/bin/env python3
"""Layer 4 Dashboard — Voice Phishing Detection.

Run: streamlit run streamlit_app.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import soundfile as sf

from src.phishing_vocabulary import PHISHING_VOCABULARY
from src.dashboard.pipeline_runner import run_pipeline
from src.dashboard.highlighting import render_highlighted_transcript, CSS
from src.dashboard.components import (
    render_verdict_banner,
    render_class_probs_chart,
    render_audio_player,
    render_keyword_chips,
    render_driver_chips,
)
from resample import resample_to_16khz


st.set_page_config(
    page_title="Voice Phishing Detection",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Voice Phishing Detection")
st.caption("Layers 1 + 2 + 3 — multimodal fraud classifier")

uploaded = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded is None:
    st.info("Upload a call recording to get started.")
    st.stop()

# Write upload to temp file
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

try:
    # Validate it's audio
    try:
        info = sf.info(tmp_path)
    except Exception:
        st.error("Could not read the file as audio. Please upload a valid WAV.")
        st.stop()

    if info.duration < 1.0:
        st.error("Audio too short — need at least 1 second of speech.")
        st.stop()

    # Resample if needed
    audio_path = tmp_path
    if info.samplerate != 16000 or info.channels != 1:
        resampled = tmp_path.replace(".wav", "_16k.wav")
        resample_to_16khz(tmp_path, resampled)
        audio_path = resampled

    # Run pipeline
    with st.spinner("Running pipeline… (this takes ~20–30s if ASR is cold)"):
        result = run_pipeline(audio_path)

    # Error handling
    if result.error == "models_not_trained":
        st.error(
            "Layer 3 models not found. Train them via `notebooks/train_colab.ipynb` "
            "and drop the resulting `models/*.joblib` files into the repo root. "
            "See `docs/HANDOFF.md` for step-by-step instructions."
        )
        st.stop()

    if result.error:
        st.error(f"Pipeline failed: {result.error}")
        if "vosk" in (result.error or "").lower() or "model" in (result.error or "").lower():
            st.info("Vosk ASR model missing? Run: `cd models && wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip && unzip *.zip`")
        st.stop()

    l1, l2, l3 = result.layer1, result.layer2, result.layer3

    # ── Verdict banner ──────────────────────────────────────────
    render_verdict_banner(l3)

    # ── Two-column: chart + audio ────────────────────────────────
    col_chart, col_meta = st.columns([2, 1])
    with col_chart:
        render_class_probs_chart(l3)
    with col_meta:
        st.markdown("**File info**")
        st.markdown(f"- Duration: **{l1.metadata.duration_seconds:.1f}s**")
        st.markdown(f"- Sample rate: **{l1.metadata.sample_rate_hz} Hz**")
        st.markdown(f"- Words: **{l2.transcript.word_count}**")
        st.markdown(f"- Pauses: **{l2.features.pause_count}**")
        st.markdown(f"- Speech rate: **{l2.features.speech_rate:.2f} syl/s**")
        st.markdown(f"- Pitch: **{l2.features.pitch_mean:.0f} Hz**")
        st.markdown("**Audio**")
        render_audio_player(audio_path)

    st.divider()

    # ── Transcript ───────────────────────────────────────────────
    st.markdown("### Transcript")
    segments = [
        {"text": seg.text, "start_time": seg.start_time,
         "end_time": seg.end_time, "confidence": seg.confidence}
        for seg in l2.transcript.segments
    ] if l2.transcript.segments else [{"text": w, "start_time": 0.0, "end_time": 0.0, "confidence": 1.0}
                                       for w in l2.transcript.full_text.split()]

    html = render_highlighted_transcript(segments, l3.top_keywords, PHISHING_VOCABULARY)

    if html:
        st.markdown(html, unsafe_allow_html=True)
        st.caption(
            "🔴 model + curated &nbsp;|&nbsp; 🟡 model only &nbsp;|&nbsp; "
            "<u>underline</u> curated only",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("_(no transcript)_")

    st.divider()

    # ── Chips ────────────────────────────────────────────────────
    chip_col1, chip_col2 = st.columns(2)
    with chip_col1:
        render_keyword_chips(l3)
    with chip_col2:
        render_driver_chips(l3)

    # ── Per-modality breakdown ───────────────────────────────────
    with st.expander("Per-modality detail"):
        m = l3.per_modality
        mc1, mc2, mc3 = st.columns(3)
        for col, key in zip([mc1, mc2, mc3], ["text", "acoustic", "fused"]):
            probs = m.get(key, {})
            top = max(probs, key=probs.get) if probs else "—"
            conf = probs.get(top, 0.0)
            col.metric(label=key.capitalize(), value=top.replace("_", " "), delta=f"{conf:.0%}")

finally:
    try:
        os.unlink(tmp_path)
        resampled_path = tmp_path.replace(".wav", "_16k.wav")
        if os.path.exists(resampled_path):
            os.unlink(resampled_path)
    except Exception:
        pass
