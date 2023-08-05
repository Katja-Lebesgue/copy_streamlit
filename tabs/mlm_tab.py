import streamlit as st
from enum import Enum
import os
from dotenv import load_dotenv
import requests
import pandas as pd
from loguru import logger

load_dotenv()

API_HOST = os.getenv("API_HOST")


class MLModel(str, Enum):
    wiki = "wiki"
    ad = "ad"


def mlm_tab():
    masked_text = st.text_area(
        label="Ad copy",
        height=1,
        value="Turn heads this ??? with pieces from the same ??? as Reformation, Zimmermann, Faithfull the Brand and more, for ??? 50-80% ???. Buy now ??? ⭐⭐⭐⭐???",
        key="mlm_text",
        help="Enter ad copy and replace words you want our models to predict with three question marks (???).",
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Wiki Model")
        st.info(
            "This model was trained on Wikipedia corpus and has a generally well understanding of English language."
        )
        mlm_column(model_name=MLModel.wiki, masked_text=masked_text, unmasked_words_color="Crimson")

    with col2:
        st.subheader("Lebesgue AdModel")
        st.info(
            "This model was trained on Lebesgue users data and is therefore more ad-specialized.",
        )
        mlm_column(model_name=MLModel.ad, masked_text=masked_text, unmasked_words_color="DarkCyan")


@st.cache_data
def infer_mlm(masked_text: str, model_name: MLModel):
    mlm_request_params = {"masked_text": masked_text, "model_name": model_name}
    mlm_response = requests.post(f"{API_HOST}/mlm", params=mlm_request_params).json()
    return mlm_response


def mlm_column(model_name: MLModel, masked_text: str, unmasked_words_color: str = "red"):
    mlm_response = infer_mlm(masked_text=masked_text, model_name=model_name)

    mask_predictions = list(mlm_response.values())

    unmasked_text = ""
    previous_index = -3

    for pred in mask_predictions:
        new_index = previous_index + 3 + masked_text[previous_index + 3 :].find("???")
        unmasked_text = unmasked_text + masked_text[previous_index + 3 : new_index + 3].replace(
            "???", f"<span style='color:{unmasked_words_color}; font-weight:bold'>{pred[0]}</span>"
        )
        previous_index = new_index

    unmasked_text = unmasked_text + masked_text[previous_index + 3 :]

    unmasked_text = f"<p> {unmasked_text} </p>"

    st.caption("Word predictions")
    st.markdown(unmasked_text, unsafe_allow_html=True)

    st.caption("Most likely words")

    df = pd.DataFrame(mlm_response).T
    df.columns = map(lambda x: f"#{x}", list(range(1, len(df.columns) + 1)))
    df.index = map(lambda x: f"??? {x}", list(range(1, len(df.index) + 1)))

    st.dataframe(df)
