import os
from uuid import uuid4

import requests
import streamlit as st

API_URL = os.getenv("EATWISE_API_URL", "http://localhost:8000").rstrip("/")

st.title("EatWise")
user_id = st.text_input("User ID", value="u1").strip()

if st.button("Get recommendations", disabled=not user_id):
    st.session_state.pop("recommendation_round", None)
    try:
        response = requests.get(f"{API_URL}/recommend", params={"user_id": user_id}, timeout=5)
        response.raise_for_status()
        st.session_state["recommendation_round"] = {
            "session_id": uuid4().hex,
            "result": response.json(),
            "outside_option_sent": False,
        }
    except requests.RequestException as error:
        st.error(f"Could not load recommendations. {error}")

round_state = st.session_state.get("recommendation_round")
if round_state and round_state["result"]["user_id"] == user_id:
    result = round_state["result"]
    if result["mock"]:
        st.info("These recommendations use mock data.")

    recipes = result["recommendations"]
    for recipe in recipes:
        with st.container(border=True):
            st.subheader(recipe["title"])
            st.caption(f"Expected consumed enrichment: {recipe['score']:.2f}")

    if st.button("I ate something else", disabled=round_state["outside_option_sent"]):
        try:
            response = requests.post(
                f"{API_URL}/log_interaction",
                json={
                    "user_id": user_id,
                    "session_id": round_state["session_id"],
                    "shown_recipe_ids": [recipe["recipe_id"] for recipe in recipes],
                    "chosen_recipe_id": "outside_option",
                },
                timeout=5,
            )
            response.raise_for_status()
            round_state["outside_option_sent"] = True
        except requests.RequestException as error:
            st.error(f"Could not submit the outside option. {error}")

    if round_state["outside_option_sent"]:
        message = "Outside option sent to the mock API." if result["mock"] else "Outside option submitted."
        st.success(message)
