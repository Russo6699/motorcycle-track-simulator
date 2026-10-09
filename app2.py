"""Streamlit entry point for the motorcycle track simulator.

The app itself is a browser application (index.html + motosim.js): map, real elevation data,
3D road, ideal-line optimisation and ride-along. Streamlit only serves it full-page, so the
existing Streamlit Cloud link keeps working. index.html can also be opened directly or served
by GitHub Pages.
"""
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="3D Motorcycle Track Simulator", page_icon="🏍️",
                   layout="wide", initial_sidebar_state="collapsed")

# hide Streamlit chrome so the app gets the whole page
st.markdown(
    """
    <style>
      #MainMenu, header, footer {visibility: hidden;}
      .block-container {padding: 0 !important; max-width: 100% !important;}
      iframe {display: block; border: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

HERE = Path(__file__).parent


@st.cache_data
def load_app() -> str:
    html = (HERE / "index.html").read_text(encoding="utf-8")
    engine = (HERE / "motosim.js").read_text(encoding="utf-8")
    # inline the physics engine: the component iframe cannot load sibling files
    return html.replace('<script src="motosim.js"></script>', "<script>\n" + engine + "\n</script>")


components.html(load_app(), height=1050, scrolling=True)
