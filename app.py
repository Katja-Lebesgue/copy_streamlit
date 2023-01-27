import streamlit as st
from enum import Enum
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from loguru import logger

from utils.enum import get_enum_values

load_dotenv()

API_PASSWORD = os.getenv("API_PASSWORD")
API_HOST = os.getenv("API_HOST")

logger.debug(f"API HOST: {API_HOST}")

logger.debug(f"API PASS: {API_PASSWORD}")


class Target(str, Enum):
    acquisition = "acquisition"
    remarketing = "remarketing"


class CreativeType(str, Enum):
    carousel = "carousel"
    dynamic = "dynamic"
    image = "image"
    video = "video"
    unknown = "unknown"


st.set_page_config(layout="wide")


@st.cache
def infer_ctr(request_json: dict) -> tuple:
    request_kwargs = {"json": request_json, "params": {"password": API_PASSWORD}}
    classification_result = requests.post(f"{API_HOST}/classification", **request_kwargs).json()[0]
    regression_result = requests.post(f"{API_HOST}/regression", **request_kwargs).json()[0]
    return classification_result, regression_result


def ctr_pred():

    st.subheader("CTR prediction")

    text = st.text_area(
        label="Ad copy",
        height=1,
        value="Turn heads this summer with pieces from the same manufacturers as Reformation, Zimmermann, Faithfull the Brand and more, for 50-80% less.",
        key="ctr_text",
    )

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

    request_json = [{"text": text, "aov": aov, "target": target, "creative_type": creative_type}]

    ctr_interval, ctr_value = infer_ctr(request_json=request_json)

    df = pd.DataFrame([ctr_value, ctr_interval], index=["value", "interval"], columns=["ctr"])

    df_style = df.style
    df_style.hide(axis="columns")

    df_style.format(formatter=lambda x: "{:,.2f}%".format(x * 100), subset=("value", df.columns))

    df_style.format(
        formatter=lambda x: str(["{:,.2f}%".format(num * 100) for num in x]), subset=("interval", df.columns)
    )

    st.table(df_style)


@st.cache
def infer_mlm(masked_text: str):
    mlm_request_params = {"masked_text": masked_text, "password": API_PASSWORD}
    mlm_response = requests.post(f"{API_HOST}/mlm", params=mlm_request_params).json()
    return mlm_response


def mlm():

    st.subheader("Predicting masked words")

    masked_text = st.text_area(
        label="Ad copy",
        height=1,
        value="Turn heads this ??? with pieces from the same ??? as Reformation, Zimmermann, Faithfull the Brand and more, for ??? 50-80% ???. Buy now ??? ⭐⭐⭐⭐???",
        key="mlm_text",
    )

    mlm_response = infer_mlm(masked_text=masked_text)

    mask_predictions = list(mlm_response.values())

    unmasked_text = ""
    previous_index = -3

    for pred in mask_predictions:
        new_index = previous_index + 3 + masked_text[previous_index + 3 :].find("???")
        logger.debug(new_index)
        unmasked_text = unmasked_text + masked_text[previous_index + 3 : new_index + 3].replace(
            "???", f"<span style='color:red; font-weight:bold'>{pred[0]}</span>"
        )
        previous_index = new_index

    unmasked_text = unmasked_text + masked_text[previous_index + 3 :]

    unmasked_text = f"<p> {unmasked_text} </p>"

    st.caption("Unmasked text")
    st.markdown(unmasked_text, unsafe_allow_html=True)

    st.caption("Most likely words")

    df = pd.DataFrame(mlm_response).T

    st.table(df)


# main
st.markdown("<h1 style='text-align: center;'>⭐San-Jin Engine⭐</h1>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    ctr_pred()

with col2:
    mlm()
