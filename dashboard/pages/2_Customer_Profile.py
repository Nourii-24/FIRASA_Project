import streamlit as st

st.set_page_config(page_title="Firasa | Customer Profile", page_icon="\U0001F985", layout="wide")
st.markdown(
    """
    <style>
      html, body, .stApp { background: #0D2B1E !important; color: #F3EFE3 !important; }
      [data-testid="stSidebar"] { background: #081C13 !important; }
      [data-testid="stSidebar"] * { color: #F3EFE3 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Customer profile")
st.info("Placeholder page. Build the single-customer risk breakdown here.")
