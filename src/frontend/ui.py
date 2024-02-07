import pandas as pd
import requests
import streamlit as st


@st.cache_data
def get_highlights(video_id):
    result_json = requests.get(f"http://127.0.0.1:8000/highlights/{video_id}").json()
    result_df = pd.json_normalize(result_json)
    return result_df


video_id = st.text_input("Enter Twitch VOD ID")

if st.button("Extract highlights"):
    highlights = get_highlights(video_id)
    st.write(highlights)
