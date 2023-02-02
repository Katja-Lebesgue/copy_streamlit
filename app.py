import streamlit as st
from streamlit_option_menu import option_menu
from enum import Enum
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from loguru import logger

from authenticate import authenticate
from utils.enum import get_enum_values
from tabs.ctr_tab import ctr_tab
from tabs.mlm_tab import mlm_tab

load_dotenv()

API_HOST = os.getenv("API_HOST")
st.set_page_config(layout="wide")

authenticator = authenticate()

if st.session_state["authentication_status"]:
    authenticator.logout("Logout", "sidebar")

    with st.sidebar:

        st.markdown("<h1 style='text-align: center;'>⭐SanJin EnGin⭐</h1>", unsafe_allow_html=True)

        main_tab = option_menu(
            menu_title="Main menu",
            options=["CTR", "Masked language"],
            menu_icon="menu-app",
            icons=["mouse", "mask"],
            default_index=0,
        )

    if main_tab == "CTR":
        ctr_tab()

    if main_tab == "Masked language":
        mlm_tab()
