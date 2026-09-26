import base64
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Firasa | Credit Risk Intelligence",
    page_icon="\U0001F985",
    layout="centered",
)

ASSETS_DIR = Path(__file__).parent / "assets"
DATA_DIR = Path(__file__).parent / "data"


def load_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def get_customer_count(default: int = 8297) -> int:
    """Pull the real customer count from the scored dataset if it's there,
    otherwise fall back to the known default so the page still renders."""
    scores_path = DATA_DIR / "customer_scores_with_shap.csv"
    if scores_path.exists():
        try:
            return int(len(pd.read_csv(scores_path, usecols=[0])))
        except Exception:
            return default
    return default


# Hide Streamlit's default chrome so the hero reads as a real landing page,
# and style st.page_link so the three nav links match the card's gold-pill look.
st.markdown(
    """
    <style>
      #MainMenu, header, footer { visibility: hidden; }

      html, body, .stApp {
        background: #0D2B1E !important;
      }
      [data-testid="stSidebar"] {
        background: #081C13 !important;
      }
      [data-testid="stSidebar"] * {
        color: #F3EFE3 !important;
      }
      .block-container { padding-top: 1rem; padding-bottom: 2rem; max-width: 640px; }

      div[data-testid="stPageLink"] {
        margin: 0 !important;
      }
      div[data-testid="stPageLink"] a {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        padding: 12px 14px;
        border-radius: 999px;
        border: 1px solid #D4AF37;
        background: linear-gradient(180deg, rgba(212,175,55,0.16), rgba(212,175,55,0.05)) !important;
        color: #E9CB6F !important;
        font-family: 'Inter', sans-serif;
        font-size: 13px;
        font-weight: 600;
        text-decoration: none !important;
        transition: background 0.2s ease, transform 0.15s ease;
      }
      div[data-testid="stPageLink"] a:hover {
        background: linear-gradient(180deg, rgba(212,175,55,0.28), rgba(212,175,55,0.10)) !important;
        transform: translateY(-1px);
      }
      div[data-testid="stPageLink"] a p {
        color: #E9CB6F !important;
        font-weight: 600 !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero: brand, animated card, live customer count ---
hero_template = (ASSETS_DIR / "hero.html").read_text(encoding="utf-8")
hero_html = (
    hero_template
    .replace("__FRONT_B64__", load_b64(ASSETS_DIR / "card_front.png"))
    .replace("__BACK_B64__", load_b64(ASSETS_DIR / "card_back.png"))
    .replace("__CUSTOMER_COUNT__", str(get_customer_count()))
)
components.html(hero_html, height=560, scrolling=False)

# --- Navigation: three gold-pill links, evenly spaced ---
col1, col2, col3 = st.columns(3, gap="medium")
with col1:
    st.page_link("pages/1_Radar.py", label="Radar View")
with col2:
    st.page_link("pages/2_Customer_Profile.py", label="Customer Profile")
with col3:
    st.page_link("pages/3_Manager_View.py", label="Manager Dashboard")
