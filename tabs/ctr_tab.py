import streamlit as st
from streamlit.components import v1 as components
from streamlit_option_menu import option_menu
from enum import Enum
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from loguru import logger
import numpy as np
import shap
from matplotlib import pyplot as plt

import streamlit.components.v1 as components
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils import capture

load_dotenv()

from utils.enum import get_enum_values

API_HOST = os.getenv("API_HOST")


model_frontend_format = {
    "classification3": "CTR interval",
    "classification3_text": "CTR interval based on text only",
    "regression": "CTR value",
}


class CTRModel(str, Enum):
    classification3 = "classification3"
    regression = "regression"
    classification3_text = "classification3_text"


class Target(str, Enum):
    acquisition = "acquisition"
    remarketing = "remarketing"


class CreativeType(str, Enum):
    carousel = "carousel"
    dynamic = "dynamic"
    image = "image"
    video = "video"
    unknown = "unknown"


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def ctr_tab():

    col1, col2 = st.columns(2)

    with col1:
        text = st.text_area(
            label="Ad copy",
            height=215,
            value="Turn heads this summer with pieces from the same manufacturers as Reformation, Zimmermann, Faithfull the Brand and more, for 50-80% less.",
            key="ctr_text",
        )

    with col2:
        aov = st.number_input(label="Average order value (in USD)", step=10, value=200)

        target = st.selectbox(
            "Target",
            tuple(get_enum_values(Target, sort=True)),
            format_func=lambda x: x,
            key="target",
        )

        creative_type = st.selectbox(
            "Creative type",
            tuple(get_enum_values(CreativeType, sort=True)),
            format_func=lambda x: x,
            key="creative_type",
        )

    request_json = {"text": text, "aov": aov, "target": target, "creative_type": creative_type}

    st.subheader("CTR prediction")
    ctr_prediction(request_json=request_json)

    st.subheader("Word impacts")
    st.markdown("<p>See how each word impacts final prediction in every model.</p>", unsafe_allow_html=True)

    model_type = st.selectbox(
        "Model",
        tuple(get_enum_values(CTRModel, sort=True)),
        format_func=lambda x: model_frontend_format[x],
        key="model_type",
    )

    get_shap = st.button(label="Analyse")

    if get_shap:
        shap_analysis(request_json=request_json, model_type=model_type)


def ctr_prediction(request_json: dict):

    result_dict = {}

    for model_type in get_enum_values(CTRModel, sort=True):
        # for model_type in [CTRModel.classification3_text]:
        result = infer_ctr(request_json=request_json, model_type=model_type)
        result_dict[model_type] = [result]

    df = pd.DataFrame(result_dict, index=["ctr (%)"])
    df.columns = [model_frontend_format[col] for col in df.columns]
    df = df.T

    df_style = df.style

    def format_func(x):
        if type(x) == list:
            return [round(a * 100, 2) for a in x]
        else:
            return round(x * 100, 2)

    df_style.format(formatter=format_func)

    st.dataframe(df_style)


# @st.cache
def infer_ctr(request_json: dict, model_type: CTRModel) -> tuple:
    request_kwargs = {"json": [request_json], "params": {"model_type": model_type}}
    result = requests.post(f"{API_HOST}/ctr", **request_kwargs).json()[0]
    return result


def shap_analysis(request_json: dict, model_type: CTRModel = CTRModel.classification3):

    shap_kwargs = get_shap_values(request_json=request_json, model_type=model_type).copy()
    shap_kwargs = unpack_shap_kwargs(shap_kwargs=shap_kwargs)
    shap_values = shap.Explanation(**shap_kwargs)
    st_plot_text_shap(shap_val=shap_values, height=400)


# @st.cache
def get_shap_values(request_json: dict, model_type: CTRModel):
    request_kwargs = {"json": request_json, "params": {"model_type": model_type}}
    result = requests.post(f"{API_HOST}/shap", **request_kwargs).json()
    return result


def unpack_shap_kwargs(shap_kwargs: dict):
    unpacked = shap_kwargs.copy()
    unpacked.update({k: np.array(v) for k, v in shap_kwargs.items() if k in ["values", "base_values"]})
    unpacked["data"] = (np.array(shap_kwargs["data"]),)
    return unpacked


def st_plot_text_shap(shap_val, height=None):
    InteractiveShell().instance()
    with capture.capture_output() as cap:
        shap.plots.text(shap_val)
    components.html(cap.outputs[0].data["text/html"], height=height, scrolling=True)


def is_regression(model_type: CTRModel):
    return "regression" in model_type
