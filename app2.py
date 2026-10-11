"""Streamlit entry point for the motorcycle track simulator.

The app itself is a browser application (index.html + motosim.js + satcorridor.js + roadfit.py): map, real elevation data,
3D road, ideal-line optimisation and ride-along. Streamlit only serves it full-page, so the
existing Streamlit Cloud link keeps working. index.html can also be opened directly or served
by GitHub Pages.
"""
import re
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


FILES = ("index.html", "motosim.js", "satcorridor.js", "roadfit.py")


@st.cache_data
def load_app(stamp: tuple) -> str:
    # `stamp` (size + modification time of every file) is part of the cache key, so a new version of any
    # of the files is served as soon as it is deployed instead of the cached old page
    html = (HERE / "index.html").read_text(encoding="utf-8")
    # the component iframe cannot load sibling files, so the scripts are inlined
    for name in ("motosim.js", "satcorridor.js"):
        code = (HERE / name).read_text(encoding="utf-8")
        html = html.replace(f'<script src="{name}"></script>', "<script>\n" + code + "\n</script>")
    # the satellite road-extraction step runs roadfit.py in the browser (Pyodide); embed the current file
    roadfit = (HERE / "roadfit.py").read_text(encoding="utf-8")
    html = re.sub(r'(<script type="text/x-python" id="roadfit-py">\n).*?(</script>)',
                  lambda m: m.group(1) + roadfit + m.group(2), html, count=1, flags=re.S)
    return html


stamp = tuple((f, (HERE / f).stat().st_size, (HERE / f).stat().st_mtime_ns) for f in FILES)
components.html(load_app(stamp), height=1050, scrolling=True)
