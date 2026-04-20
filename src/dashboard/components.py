import streamlit as st
import plotly.graph_objects as go
from src.layer3_constants import CLASS_LABELS


_VERDICT_COLOURS = {
    "legitimate": "#21c55d",
    "customer_service": "#6b7280",
}
_FRAUD_COLOUR = "#ef4444"


def render_verdict_banner(l3) -> None:
    colour = _VERDICT_COLOURS.get(l3.label, _FRAUD_COLOUR)
    icon = "✅" if l3.label == "legitimate" else "⚠️"
    st.markdown(
        f"""
        <div style="
            background:{colour};color:white;border-radius:8px;
            padding:16px 24px;font-size:1.4rem;font-weight:700;
            text-align:center;margin-bottom:8px;">
            {icon} {l3.label.upper().replace("_", " ")} &nbsp;·&nbsp; {l3.confidence:.0%} confidence
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_class_probs_chart(l3) -> None:
    probs = [l3.class_probabilities.get(lbl, 0.0) for lbl in CLASS_LABELS]
    colours = [
        _VERDICT_COLOURS.get(lbl, _FRAUD_COLOUR) if lbl == l3.label else "#d1d5db"
        for lbl in CLASS_LABELS
    ]
    fig = go.Figure(go.Bar(
        x=CLASS_LABELS,
        y=probs,
        marker_color=colours,
        text=[f"{p:.1%}" for p in probs],
        textposition="outside",
    ))
    fig.update_layout(
        title="Class Probabilities",
        yaxis=dict(range=[0, 1.15], tickformat=".0%"),
        xaxis=dict(tickangle=-30),
        height=320,
        margin=dict(t=40, b=60, l=20, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_audio_player(path: str) -> None:
    with open(path, "rb") as f:
        st.audio(f.read(), format="audio/wav")


def render_keyword_chips(l3) -> None:
    if not l3.top_keywords:
        return
    st.markdown("**Top keywords**")
    st.markdown(
        " ".join(
            f'<span style="background:#ffd700;color:black;border-radius:12px;'
            f'padding:3px 10px;margin:2px;display:inline-block;">{kw}</span>'
            for kw in l3.top_keywords
        ),
        unsafe_allow_html=True,
    )


def render_driver_chips(l3) -> None:
    if not l3.top_acoustic_drivers:
        return
    st.markdown("**Top acoustic drivers**")
    st.markdown(
        " ".join(
            f'<span style="background:#e0e7ff;color:#3730a3;border-radius:12px;'
            f'padding:3px 10px;margin:2px;display:inline-block;">{d}</span>'
            for d in l3.top_acoustic_drivers
        ),
        unsafe_allow_html=True,
    )
