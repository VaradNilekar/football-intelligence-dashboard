import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# Page config — this MUST be the first Streamlit command in the script

st.set_page_config(
    page_title="Football Player Value Predictor",
    page_icon="⚽",
    layout="centered"
)


# Load the trained model once and cache it.
# @st.cache_resource tells Streamlit: "run this function once, then reuse
# the result across every rerun" — without it, the model would reload from
# disk every single time a user moves a slider, which is slow and wasteful.

@st.cache_resource
def load_model():
    model = joblib.load("value_model.pkl")
    with open("feature_columns.json") as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_model()

# Header

st.title("⚽ Football Player Value Predictor")
st.write(
    "Enter a player's attributes and get a predicted market value, using a "
    "Random Forest model trained on 3,467 players from Europe's top 5 leagues "
    "(FIFA 24 data). The model relies almost entirely on **overall rating** and "
    "**potential** — wage data turned out to add no real predictive power once "
    "those two are known. [Read the full analysis on GitHub]"
    "(https://github.com/VaradNilekar/football-intelligence-dashboard)."
)

st.divider()

# Input widgets, laid out in two columns for a tidier form.
# st.columns(2) splits the page into two side-by-side areas; anything called
# inside "with col1:" renders in the left column, and so on.

col1, col2 = st.columns(2)

with col1:
    st.subheader("Core ratings")
    overall = st.slider("Overall rating", 40, 99, 75)
    potential = st.slider("Potential", 40, 99, 80)
    age = st.slider("Age", 16, 42, 24)
    international_reputation = st.slider("International reputation", 1, 5, 2)
    position_group = st.selectbox(
        "Position", ["Defender", "Forward", "Goalkeeper", "Midfielder"]
    )

with col2:
    st.subheader("Attributes")
    pace = st.slider("Pace", 0, 99, 70)
    shooting = st.slider("Shooting", 0, 99, 60)
    passing = st.slider("Passing", 0, 99, 65)
    dribbling = st.slider("Dribbling", 0, 99, 68)
    defending = st.slider("Defending", 0, 99, 55)
    physic = st.slider("Physical", 0, 99, 70)
    weak_foot = st.slider("Weak foot (stars)", 1, 5, 3)
    skill_moves = st.slider("Skill moves (stars)", 1, 5, 2)

st.divider()

# Build the input row EXACTLY matching the training feature columns.
# The model has no idea what "Forward" means as text — during training we
# one-hot encoded position_group into pos_Forward / pos_Goalkeeper /
# pos_Midfielder (Defender was the dropped baseline category). We have to
# recreate that same encoding here, in the same column order, or the model
# will silently misread which number belongs to which feature.
def build_input_row(overall, potential, age, international_reputation,
                     pace, shooting, passing, dribbling, defending, physic,
                     weak_foot, skill_moves, position_group, feature_columns):
    row = {
        "overall": overall,
        "potential": potential,
        "age": age,
        "international_reputation": international_reputation,
        "pace": pace,
        "shooting": shooting,
        "passing": passing,
        "dribbling": dribbling,
        "defending": defending,
        "physic": physic,
        "weak_foot": weak_foot,
        "skill_moves": skill_moves,
        "pos_Forward": 1 if position_group == "Forward" else 0,
        "pos_Goalkeeper": 1 if position_group == "Goalkeeper" else 0,
        "pos_Midfielder": 1 if position_group == "Midfielder" else 0,
    }
    # Reorder to match training exactly, as a single-row DataFrame
    return pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)


# Predict button. Every widget interaction reruns the whole script top to
# bottom — st.button returns True only on the run where it was just clicked,
# so the prediction block only fires then.

if st.button("Predict Market Value", type="primary"):
    X_input = build_input_row(
        overall, potential, age, international_reputation,
        pace, shooting, passing, dribbling, defending, physic,
        weak_foot, skill_moves, position_group, feature_columns
    )

    # Model was trained on log1p(value_eur), so undo that transform to get euros back
    pred_log = model.predict(X_input)[0]
    pred_eur = np.expm1(pred_log)

    st.metric(label="Predicted Market Value", value=f"€{pred_eur:,.0f}")

    if pred_eur > 50_000_000:
        st.caption("💎 World-class superstar territory.")
    elif pred_eur > 10_000_000:
        st.caption("⭐ Established top-flight talent.")
    elif pred_eur > 1_000_000:
        st.caption("👍 Solid squad player.")
    else:
        st.caption("🌱 Developing / squad depth player.")

st.divider()
st.caption(
    "Model: Random Forest Regressor · R² ≈ 0.97 on held-out test data · "
    "Trained without wage_eur (see project README for why)."
)
