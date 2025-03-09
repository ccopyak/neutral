import streamlit as st
import pandas as pd
import joblib

# Load trained Logistic Regression model
model = joblib.load("log_reg_model.pkl")

# Load team dataset
df = pd.read_excel("data.xlsx")

# Streamlit UI
st.title("🏀 Basketball Game Predictor (Neutral Court)")

# Select teams
team_names = df["Team"].dropna().unique().tolist()

home_team = st.selectbox("Select Team 1", team_names)
away_team = st.selectbox("Select Team 2", team_names)

if st.button("Predict Winner"):
    # Extract team stats
    home_team_info = df[df["Team"] == home_team]
    away_team_info = df[df["Team"] == away_team]

    if home_team_info.empty or away_team_info.empty:
        st.error("Team data not found!")
    else:
        # Prepare input features
        home_team_data = home_team_info[['NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck']].values.flatten()
        away_team_data = away_team_info[['NetRtg', 'ORtg', 'DRtg', 'AdjT', 'Luck']].values.flatten()

        # Function to make prediction
        def predict_winner(home_team_data, away_team_data):
            features = {
                'Home NetRtg': home_team_data[0],
                'Home Ortg': home_team_data[1],
                'Home DRtg': home_team_data[2],
                'Home AdjT': home_team_data[3],
                'Home Luck': home_team_data[4],
                'Away NetRtg': away_team_data[0],
                'Away Ortg': away_team_data[1],
                'Away DRtg': away_team_data[2],
                'Away AdjT': away_team_data[3],
                'Away Luck': away_team_data[4]
            }
            features_df = pd.DataFrame([features])
            prediction = model.predict(features_df)[0]
            probability = model.predict_proba(features_df)[:, 1][0]  # Prob of "home" winning
            return prediction, probability

        # Run prediction normally
        _, prob1 = predict_winner(home_team_data, away_team_data)

        # Run prediction with swapped home/away
        _, prob2 = predict_winner(away_team_data, home_team_data)

        # Average probabilities
        avg_prob = (prob1 + (1 - prob2)) / 2

        # Determine final neutral court winner
        if avg_prob > 0.5:
            st.success(f"🏆 **Prediction: {home_team} Wins on a Neutral Court!**")
            st.info(f"Win Probability: {avg_prob * 100:.1f}%")
        else:
            st.success(f"🏆 **Prediction: {away_team} Wins on a Neutral Court!**")
            st.info(f"Win Probability: {(1 - avg_prob) * 100:.1f}%")
