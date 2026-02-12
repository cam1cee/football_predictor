import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from io import BytesIO
import requests
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Football Prediction App",
    page_icon="⚽",
    layout="wide"
)

# League mappings for football-data.co.uk
LEAGUE_MAPPINGS = {
    'Premier League': {'code': 'E0', 'seasons': ['2324', '2425', '2526']},
    'La Liga': {'code': 'SP1', 'seasons': ['2324', '2425', '2526']},
    'Serie A': {'code': 'I1', 'seasons': ['2324', '2425', '2526']},
    'Serie B': {'code': 'I2', 'seasons': ['2324', '2425', '2526']},
    'Bundesliga 2': {'code': 'D2', 'seasons': ['2324', '2425', '2526']},
    'Swiss Super League': {'code': 'SWZ', 'seasons': ['2324', '2425', '2526']},
    'J-League': {'code': 'J1', 'seasons': ['2324', '2425', '2526']}
}

@st.cache_data(ttl=3600)
def load_data_from_url(league_code, season):
    """Load data from football-data.co.uk"""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
        df = pd.read_csv(url, encoding='latin1')
        return df
    except Exception as e:
        st.warning(f"Could not load data for {league_code} season {season}: {str(e)}")
        return None

@st.cache_data
def load_all_leagues_data():
    """Load data for all leagues and seasons"""
    all_data = []
    
    for league_name, league_info in LEAGUE_MAPPINGS.items():
        for season in league_info['seasons']:
            df = load_data_from_url(league_info['code'], season)
            if df is not None and not df.empty:
                df['League'] = league_name
                df['Season'] = season
                all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def prepare_features(df):
    """Prepare features for model training"""
    # Keep only necessary columns
    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League', 'Season']
    
    # Check which columns exist
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()
    
    # Drop rows with missing values
    df = df.dropna(subset=['FTHG', 'FTAG', 'FTR'])
    
    # Calculate team statistics
    team_stats = calculate_team_stats(df)
    
    # Merge statistics
    df = df.merge(team_stats, left_on='HomeTeam', right_index=True, suffixes=('', '_home'))
    df = df.merge(team_stats, left_on='AwayTeam', right_index=True, suffixes=('', '_away'))
    
    return df, team_stats

def calculate_team_stats(df):
    """Calculate team statistics for predictions"""
    stats_list = []
    
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        # Home games
        home_games = df[df['HomeTeam'] == team]
        # Away games
        away_games = df[df['AwayTeam'] == team]
        
        stats = {
            'team': team,
            'home_goals_scored': home_games['FTHG'].mean() if len(home_games) > 0 else 0,
            'home_goals_conceded': home_games['FTAG'].mean() if len(home_games) > 0 else 0,
            'away_goals_scored': away_games['FTAG'].mean() if len(away_games) > 0 else 0,
            'away_goals_conceded': away_games['FTHG'].mean() if len(away_games) > 0 else 0,
            'home_wins': (home_games['FTR'] == 'H').sum() if len(home_games) > 0 else 0,
            'away_wins': (away_games['FTR'] == 'A').sum() if len(away_games) > 0 else 0,
            'home_games': len(home_games),
            'away_games': len(away_games)
        }
        
        # Calculate win percentages
        stats['home_win_pct'] = stats['home_wins'] / stats['home_games'] if stats['home_games'] > 0 else 0
        stats['away_win_pct'] = stats['away_wins'] / stats['away_games'] if stats['away_games'] > 0 else 0
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list).set_index('team')

class FootballPredictor:
    def __init__(self):
        self.result_model = None
        self.goals_model = None
        self.team_stats = None
        
    def train(self, df):
        """Train prediction models"""
        df_prepared, self.team_stats = prepare_features(df)
        
        if len(df_prepared) < 10:
            st.error("Not enough data to train the model")
            return False
        
        # Prepare features for training
        feature_cols = [col for col in df_prepared.columns if col not in 
                       ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League', 'Season']]
        
        X = df_prepared[feature_cols].fillna(0)
        
        # Train result prediction model
        y_result = df_prepared['FTR']
        self.result_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.result_model.fit(X, y_result)
        
        # Train goals prediction model
        y_goals = df_prepared['FTHG'] + df_prepared['FTAG']
        self.goals_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.goals_model.fit(X, y_goals)
        
        return True
    
    def predict(self, home_team, away_team):
        """Make predictions for a match"""
        if self.team_stats is None or home_team not in self.team_stats.index or away_team not in self.team_stats.index:
            return None
        
        # Get team statistics
        home_stats = self.team_stats.loc[home_team]
        away_stats = self.team_stats.loc[away_team]
        
        # Create feature vector
        features = pd.DataFrame([{
            'home_goals_scored': home_stats['home_goals_scored'],
            'home_goals_conceded': home_stats['home_goals_conceded'],
            'away_goals_scored': home_stats['away_goals_scored'],
            'away_goals_conceded': home_stats['away_goals_conceded'],
            'home_wins': home_stats['home_wins'],
            'away_wins': home_stats['away_wins'],
            'home_games': home_stats['home_games'],
            'away_games': home_stats['away_games'],
            'home_win_pct': home_stats['home_win_pct'],
            'away_win_pct': home_stats['away_win_pct'],
            'home_goals_scored_away': away_stats['home_goals_scored'],
            'home_goals_conceded_away': away_stats['home_goals_conceded'],
            'away_goals_scored_away': away_stats['away_goals_scored'],
            'away_goals_conceded_away': away_stats['away_goals_conceded'],
            'home_wins_away': away_stats['home_wins'],
            'away_wins_away': away_stats['away_wins'],
            'home_games_away': away_stats['home_games'],
            'away_games_away': away_stats['away_games'],
            'home_win_pct_away': away_stats['home_win_pct'],
            'away_win_pct_away': away_stats['away_win_pct']
        }])
        
        # Make predictions
        result_probs = self.result_model.predict_proba(features)[0]
        total_goals = self.goals_model.predict(features)[0]
        
        # Get class labels
        classes = self.result_model.classes_
        
        predictions = {
            'home_win_prob': result_probs[list(classes).index('H')] if 'H' in classes else 0,
            'draw_prob': result_probs[list(classes).index('D')] if 'D' in classes else 0,
            'away_win_prob': result_probs[list(classes).index('A')] if 'A' in classes else 0,
            'expected_goals': total_goals,
            'expected_home_goals': home_stats['home_goals_scored'],
            'expected_away_goals': away_stats['away_goals_scored']
        }
        
        return predictions

def calculate_asian_handicap(home_prob, away_prob, home_goals, away_goals):
    """Calculate Asian Handicap lines"""
    goal_diff = home_goals - away_goals
    
    handicaps = []
    for line in np.arange(-3, 3.5, 0.5):
        adjusted_home_prob = home_prob if goal_diff > line else away_prob
        handicaps.append({
            'line': line,
            'home_cover_prob': adjusted_home_prob,
            'away_cover_prob': 1 - adjusted_home_prob
        })
    
    return handicaps

def calculate_goal_lines(expected_goals):
    """Calculate Over/Under goal lines"""
    lines = []
    for line in np.arange(0.5, 5.5, 0.5):
        # Simple probability estimation based on expected goals
        over_prob = max(0, min(1, (expected_goals - line) / 2 + 0.5))
        lines.append({
            'line': line,
            'over_prob': over_prob,
            'under_prob': 1 - over_prob
        })
    
    return lines

# Main App
st.title("⚽ Football Prediction App")
st.markdown("### Asian Handicap & Goal Lines Predictor")

# Sidebar
with st.sidebar:
    st.header("Settings")
    selected_leagues = st.multiselect(
        "Select Leagues",
        list(LEAGUE_MAPPINGS.keys()),
        default=list(LEAGUE_MAPPINGS.keys())
    )
    
    if st.button("Load Data & Train Model"):
        with st.spinner("Loading data and training model..."):
            st.session_state.data_loaded = True
            st.session_state.df = load_all_leagues_data()
            
            if not st.session_state.df.empty:
                # Filter by selected leagues
                st.session_state.df = st.session_state.df[st.session_state.df['League'].isin(selected_leagues)]
                
                # Train model
                st.session_state.predictor = FootballPredictor()
                success = st.session_state.predictor.train(st.session_state.df)
                
                if success:
                    st.success(f"✅ Loaded {len(st.session_state.df)} matches")
                    st.success("✅ Model trained successfully")
                else:
                    st.error("Failed to train model")
            else:
                st.error("No data could be loaded")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    # Get current season teams
    current_season_df = st.session_state.df[st.session_state.df['Season'] == '2526']
    all_teams = sorted(pd.concat([current_season_df['HomeTeam'], current_season_df['AwayTeam']]).unique())
    
    # Prediction interface
    st.header("Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("Home Team", all_teams)
    
    with col2:
        away_team = st.selectbox("Away Team", [t for t in all_teams if t != home_team])
    
    if st.button("Predict Match", type="primary"):
        predictions = st.session_state.predictor.predict(home_team, away_team)
        
        if predictions:
            st.header("Match Predictions")
            
            # Win Probabilities
            st.subheader("Win Probabilities")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Home Win", f"{predictions['home_win_prob']*100:.1f}%")
            with col2:
                st.metric("Draw", f"{predictions['draw_prob']*100:.1f}%")
            with col3:
                st.metric("Away Win", f"{predictions['away_win_prob']*100:.1f}%")
            
            # Expected Goals
            st.subheader("Expected Goals")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{home_team}", f"{predictions['expected_home_goals']:.2f}")
            with col2:
                st.metric("Total", f"{predictions['expected_goals']:.2f}")
            with col3:
                st.metric(f"{away_team}", f"{predictions['expected_away_goals']:.2f}")
            
            # Asian Handicap
            st.subheader("Asian Handicap Lines")
            handicaps = calculate_asian_handicap(
                predictions['home_win_prob'],
                predictions['away_win_prob'],
                predictions['expected_home_goals'],
                predictions['expected_away_goals']
            )
            
            handicap_df = pd.DataFrame(handicaps)
            handicap_df['Line'] = handicap_df['line'].apply(lambda x: f"{x:+.1f}")
            handicap_df['Home Cover %'] = (handicap_df['home_cover_prob'] * 100).round(1)
            handicap_df['Away Cover %'] = (handicap_df['away_cover_prob'] * 100).round(1)
            
            st.dataframe(
                handicap_df[['Line', 'Home Cover %', 'Away Cover %']],
                use_container_width=True,
                hide_index=True
            )
            
            # Goal Lines
            st.subheader("Over/Under Goal Lines")
            goal_lines = calculate_goal_lines(predictions['expected_goals'])
            
            goal_lines_df = pd.DataFrame(goal_lines)
            goal_lines_df['Line'] = goal_lines_df['line'].apply(lambda x: f"{x:.1f}")
            goal_lines_df['Over %'] = (goal_lines_df['over_prob'] * 100).round(1)
            goal_lines_df['Under %'] = (goal_lines_df['under_prob'] * 100).round(1)
            
            st.dataframe(
                goal_lines_df[['Line', 'Over %', 'Under %']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.error("Unable to make predictions. Teams may not have enough historical data.")
    
    # Display team statistics
    if st.checkbox("Show Team Statistics"):
        st.header("Team Statistics (2025-2026 Season)")
        
        if st.session_state.predictor.team_stats is not None:
            stats_display = st.session_state.predictor.team_stats.copy()
            stats_display = stats_display.round(2)
            st.dataframe(stats_display, use_container_width=True)
else:
    st.info("👈 Click 'Load Data & Train Model' in the sidebar to get started")
    
    st.markdown("""
    ### Features:
    - **Win Probabilities**: Home, Draw, Away predictions
    - **Expected Goals**: Team-specific and total match goals
    - **Asian Handicap Lines**: Coverage probabilities for various handicap lines
    - **Over/Under Lines**: Goal line probabilities
    
    ### Data Sources:
    - Historical data from football-data.co.uk
    - Seasons: 2023-2024, 2024-2025, 2025-2026
    - Leagues: Premier League, La Liga, Serie A, Serie B, Bundesliga 2, Swiss Super League, J-League
    """)

# Footer
st.markdown("---")
st.markdown("*Data from [football-data.co.uk](https://www.football-data.co.uk/)*")
