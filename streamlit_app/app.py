import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page config — this MUST be the first Streamlit command in the script
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Football Player Value Predictor",
    page_icon="⚽",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Anchor all file paths to this script's own folder. Streamlit Community
# Cloud always runs apps with the working directory set to the repo root,
# regardless of which subfolder app.py lives in — a plain relative path like
# "value_model.pkl" breaks on Cloud even though it works locally when you
# run `streamlit run app.py` from inside this folder.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(BASE_DIR, "value_model.pkl"))
    with open(os.path.join(BASE_DIR, "feature_columns.json")) as f:
        feature_columns = json.load(f)
    return model, feature_columns


@st.cache_resource
def load_explainer(_model):
    # SHAP's TreeExplainer is fast for tree ensembles like Random Forest and
    # gives an honest per-prediction breakdown of which features pushed the
    # prediction up or down — not just "which features matter on average"
    # (that's what feature_importances_ tells you) but "which features
    # mattered for THIS specific player."
    return shap.TreeExplainer(_model)


@st.cache_data
def load_player_directory():
    return pd.read_csv(os.path.join(BASE_DIR, "players_directory.csv"))


model, feature_columns = load_model()
explainer = load_explainer(model)
directory = load_player_directory()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚽ Football Player Value Predictor")
st.write(
    "Search a real player from Europe's top 5 leagues, or build a custom "
    "profile with the sliders, and get a predicted market value — plus a "
    "breakdown of exactly which attributes drove that prediction. Powered by "
    "a Random Forest trained on 3,467 players (R² = 0.98, 5-fold "
    "cross-validated). [Read the full analysis on GitHub]"
    "(https://github.com/VaradNilekar/football-intelligence-dashboard)."
)

st.divider()

mode = st.radio(
    "Choose a mode:",
    ["🔍 Search a real player", "🎚️ Build a custom player"],
    horizontal=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Build the input row EXACTLY matching the training feature columns.
# The model has no idea what "Forward" means as text — during training we
# one-hot encoded position_group into pos_Forward / pos_Goalkeeper /
# pos_Midfielder (Defender was the dropped baseline category). We recreate
# that same encoding here, in the same column order.
# ---------------------------------------------------------------------------
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
    return pd.DataFrame([[row[col] for col in feature_columns]], columns=feature_columns)


def show_shap_explanation(X_input, feature_columns):
    """Show a horizontal bar chart of which features pushed this specific
    prediction up (red) or down (blue), largest impact first."""
    shap_values = explainer(X_input)
    contributions = pd.Series(shap_values.values[0], index=feature_columns)
    contributions = contributions.reindex(contributions.abs().sort_values(ascending=False).index).head(8)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in contributions.values]
    ax.barh(contributions.index[::-1], contributions.values[::-1], color=colors[::-1])
    ax.set_xlabel("Impact on predicted value (log scale)")
    ax.set_title("What drove this prediction")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    st.pyplot(fig)
    st.caption(
        "🔴 Red bars push the predicted value up · 🔵 Blue bars pull it down. "
        "This shows the top factors for this specific player, not just overall feature importance."
    )


def value_tier_caption(pred_eur):
    if pred_eur > 50_000_000:
        st.caption("💎 World-class superstar territory.")
    elif pred_eur > 10_000_000:
        st.caption("⭐ Established top-flight talent.")
    elif pred_eur > 1_000_000:
        st.caption("👍 Solid squad player.")
    else:
        st.caption("🌱 Developing / squad depth player.")


# ---------------------------------------------------------------------------
# MODE 1: Search a real player
# ---------------------------------------------------------------------------
if mode == "🔍 Search a real player":
    selected_label = st.selectbox(
        "Search by name (type to filter):",
        options=directory["display_label"].tolist(),
        index=None,
        placeholder="e.g. Mbappé, Haaland, Bellingham...",
    )

    if selected_label:
        player = directory[directory["display_label"] == selected_label].iloc[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Position", player["position_group"])
        col2.metric("Age", int(player["age"]))
        col3.metric("Overall / Potential", f"{int(player['overall'])} / {int(player['potential'])}")

        with st.expander("See full attribute profile"):
            st.write(
                f"**Club:** {player['club_name']} · **Nationality:** {player['nationality_name']}\n\n"
                f"Pace {int(player['pace'])} · Shooting {int(player['shooting'])} · "
                f"Passing {int(player['passing'])} · Dribbling {int(player['dribbling'])} · "
                f"Defending {int(player['defending'])} · Physical {int(player['physic'])} · "
                f"Weak foot {int(player['weak_foot'])}★ · Skill moves {int(player['skill_moves'])}★ · "
                f"Int'l reputation {int(player['international_reputation'])}★"
            )

        X_input = build_input_row(
            player["overall"], player["potential"], player["age"], player["international_reputation"],
            player["pace"], player["shooting"], player["passing"], player["dribbling"],
            player["defending"], player["physic"], player["weak_foot"], player["skill_moves"],
            player["position_group"], feature_columns
        )
        pred_log = model.predict(X_input)[0]
        pred_eur = np.expm1(pred_log)
        actual_eur = player["value_eur"]
        delta_pct = ((pred_eur - actual_eur) / actual_eur) * 100

        st.divider()
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Actual Market Value", f"€{actual_eur:,.0f}")
        mcol2.metric("Model's Predicted Value", f"€{pred_eur:,.0f}", delta=f"{delta_pct:+.1f}%")
        value_tier_caption(pred_eur)

        st.divider()
        show_shap_explanation(X_input, feature_columns)

# ---------------------------------------------------------------------------
# MODE 2: Build a custom player
# ---------------------------------------------------------------------------
else:
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

    if st.button("Predict Market Value", type="primary"):
        X_input = build_input_row(
            overall, potential, age, international_reputation,
            pace, shooting, passing, dribbling, defending, physic,
            weak_foot, skill_moves, position_group, feature_columns
        )
        pred_log = model.predict(X_input)[0]
        pred_eur = np.expm1(pred_log)

        st.metric(label="Predicted Market Value", value=f"€{pred_eur:,.0f}")
        value_tier_caption(pred_eur)

        st.divider()
        show_shap_explanation(X_input, feature_columns)

st.divider()
st.caption(
    "Model: Random Forest Regressor · R² = 0.98 ± 0.008 (5-fold CV) · "
    "Trained without wage_eur (see project README for why). "
    "Known blind spots: aging veterans (32+) and goalkeepers — see README error analysis."
)
