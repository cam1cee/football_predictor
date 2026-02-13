import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Football Prediction App",
    page_icon="⚽",
    layout="wide"
)

# League mappings
LEAGUE_MAPPINGS = {
    'Premier League': {'code': 'E0', 'seasons': ['2324', '2425', '2526']},
    'La Liga': {'code': 'SP1', 'seasons': ['2324', '2425', '2526']},
    'Serie A': {'code': 'I1', 'seasons': ['2324', '2425', '2526']},
    'Serie B': {'code': 'I2', 'seasons': ['2324', '2425', '2526']},
    'Bundesliga 2': {'code': 'D2', 'seasons': ['2324', '2425', '2526']},
    'Swiss Super League': {'code': 'SWZ', 'seasons': ['2324', '2425', '2526']},
    'J-League': {'code': 'J1', 'seasons': ['2324', '2425', '2526']}
}

def poisson_probability(k, lambda_):
    """Calculate Poisson probability P(X=k) = (λ^k * e^(-λ)) / k!"""
    if lambda_ <= 0:
        return 0
    try:
        return (lambda_ ** k) * math.exp(-lambda_) / math.factorial(k)
    except:
        return 0

@st.cache_data(ttl=3600)
def load_data_from_url(league_code, season):
    """Load data from football-data.co.uk"""
    try:
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in df.columns for col in required_cols):
            return df
        return None
    except:
        return None

@st.cache_data
def load_all_leagues_data(selected_leagues):
    """Load data for all selected leagues"""
    all_data = []
    
    for league_name in selected_leagues:
        league_info = LEAGUE_MAPPINGS[league_name]
        for season in league_info['seasons']:
            df = load_data_from_url(league_info['code'], season)
            if df is not None and not df.empty:
                df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
                if len(df) > 0:
                    df['League'] = league_name
                    df['Season'] = season
                    all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def calculate_team_stats(df):
    """Calculate team statistics"""
    stats_dict = {}
    
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        if pd.isna(team):
            continue
        
        home_games = df[df['HomeTeam'] == team].copy()
        away_games = df[df['AwayTeam'] == team].copy()
        
        total_games = len(home_games) + len(away_games)
        if total_games == 0:
            continue
        
        stats_dict[team] = {
            'home_goals_scored': home_games['FTHG'].mean() if len(home_games) > 0 else 0,
            'home_goals_conceded': home_games['FTAG'].mean() if len(home_games) > 0 else 0,
            'away_goals_scored': away_games['FTAG'].mean() if len(away_games) > 0 else 0,
            'away_goals_conceded': away_games['FTHG'].mean() if len(away_games) > 0 else 0,
            'games_played': total_games,
            'home_games': len(home_games),
            'away_games': len(away_games)
        }
    
    return stats_dict

def predict_match(home_team, away_team, team_stats, df):
    """Predict match using Poisson distribution"""
    
    if home_team not in team_stats or away_team not in team_stats:
        return None
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # League averages
    all_home_goals = df['FTHG'].mean()
    all_away_goals = df['FTAG'].mean()
    
    # Attack and defense strengths
    home_attack = home_stats['home_goals_scored'] / all_home_goals if all_home_goals > 0 else 1
    home_defense = home_stats['home_goals_conceded'] / all_away_goals if all_away_goals > 0 else 1
    away_attack = away_stats['away_goals_scored'] / all_away_goals if all_away_goals > 0 else 1
    away_defense = away_stats['away_goals_conceded'] / all_home_goals if all_home_goals > 0 else 1
    
    # Expected goals
    exp_home = max(0.1, min(6.0, home_attack * away_defense * all_home_goals))
    exp_away = max(0.1, min(6.0, away_attack * home_defense * all_away_goals))
    
    # Calculate all scoreline probabilities
    max_goals = 10
    scoreline_probs = {}
    
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            prob = poisson_probability(h, exp_home) * poisson_probability(a, exp_away)
            scoreline_probs[(h, a)] = prob
    
    # Win probabilities
    home_win = sum(prob for (h, a), prob in scoreline_probs.items() if h > a)
    draw = sum(prob for (h, a), prob in scoreline_probs.items() if h == a)
    away_win = sum(prob for (h, a), prob in scoreline_probs.items() if h < a)
    
    # Normalize
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'exp_home': exp_home,
        'exp_away': exp_away,
        'exp_total': exp_home + exp_away,
        'scoreline_probs': scoreline_probs
    }

def calculate_asian_handicap(predictions):
    """Calculate Asian Handicap probabilities
    
    Asian Handicap logic:
    - Negative handicap (e.g., -1.0): Home team is FAVORITE, starts at -1 goal
      Home covers if they win by MORE than the handicap (goal_diff > abs(line))
    - Positive handicap (e.g., +1.0): Home team is UNDERDOG, starts at +1 goal  
      Home covers if they lose by LESS than the handicap (goal_diff > -line)
    - Zero handicap (0.0): Draw no bet, home covers if they win
    """
    scoreline_probs = predictions['scoreline_probs']
    handicaps = []
    
    for line in np.arange(-3.0, 3.5, 0.5):
        home_win = 0
        push = 0
        away_win = 0
        
        for (h, a), prob in scoreline_probs.items():
            goal_diff = h - a  # Actual goal difference
            
            if line < 0:
                # Negative handicap: home is favorite
                # Home covers if goal_diff > abs(line)
                if goal_diff > abs(line):
                    home_win += prob
                elif goal_diff == abs(line):
                    push += prob
                else:
                    away_win += prob
            elif line > 0:
                # Positive handicap: home is underdog
                # Home covers if goal_diff > -line
                if goal_diff > -line:
                    home_win += prob
                elif goal_diff == -line:
                    push += prob
                else:
                    away_win += prob
            else:
                # Zero handicap: draw no bet
                if goal_diff > 0:
                    home_win += prob
                elif goal_diff == 0:
                    push += prob
                else:
                    away_win += prob
        
        # Handle pushes (refund)
        home_final = home_win + (push * 0.5)
        away_final = away_win + (push * 0.5)
        
        handicaps.append({
            'line': line,
            'home': round(home_final * 100, 1),
            'away': round(away_final * 100, 1)
        })
    
    return handicaps

def calculate_goal_lines(predictions):
    """Calculate Over/Under lines"""
    scoreline_probs = predictions['scoreline_probs']
    lines = []
    
    for line in np.arange(0.5, 6.5, 0.5):
        over = 0
        
        for (h, a), prob in scoreline_probs.items():
            total_goals = h + a
            if total_goals > line:
                over += prob
        
        lines.append({
            'line': line,
            'over': round(over * 100, 1),
            'under': round((1 - over) * 100, 1)
        })
    
    return lines

def calculate_btts(predictions):
    """Calculate Both Teams to Score probability"""
    scoreline_probs = predictions['scoreline_probs']
    
    btts_yes = sum(prob for (h, a), prob in scoreline_probs.items() if h > 0 and a > 0)
    btts_no = 1 - btts_yes
    
    return {
        'yes': round(btts_yes * 100, 1),
        'no': round(btts_no * 100, 1)
    }

# Main App
st.title("⚽ Football Prediction App")
st.markdown("### Advanced Betting Predictions with Poisson Model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_leagues = st.multiselect(
        "Select Leagues",
        list(LEAGUE_MAPPINGS.keys()),
        default=['Premier League']
    )
    
    if st.button("🔄 Load Data", type="primary", use_container_width=True):
        if not selected_leagues:
            st.error("Please select at least one league")
        else:
            with st.spinner("Loading data..."):
                df = load_all_leagues_data(selected_leagues)
                
                if not df.empty:
                    st.session_state.df = df
                    st.session_state.team_stats = calculate_team_stats(df)
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} matches")
                    st.rerun()
                else:
                    st.error("Failed to load data")

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.session_state.data_loaded:
    df = st.session_state.df
    team_stats = st.session_state.team_stats
    
    # Get current season teams
    current_df = df[df['Season'] == '2526']
    if current_df.empty:
        current_df = df[df['Season'] == df['Season'].max()]
    
    all_teams = sorted(pd.concat([current_df['HomeTeam'], current_df['AwayTeam']]).dropna().unique())
    
    # Match selection
    st.header("🎯 Match Prediction")
    
    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("🏠 Home Team", all_teams)
    with col2:
        away_team = st.selectbox("✈️ Away Team", [t for t in all_teams if t != home_team])
    
    if st.button("📊 Predict", type="primary", use_container_width=True):
        predictions = predict_match(home_team, away_team, team_stats, df)
        
        if predictions:
            st.markdown("---")
            
            # Win Probabilities
            st.subheader("🏆 Match Result")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Home Win", f"{predictions['home_win']*100:.1f}%")
            with col2:
                st.metric("Draw", f"{predictions['draw']*100:.1f}%")
            with col3:
                st.metric("Away Win", f"{predictions['away_win']*100:.1f}%")
            
            # Expected Goals
            st.subheader("⚽ Expected Goals")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(home_team, f"{predictions['exp_home']:.2f}")
            with col2:
                st.metric("Total", f"{predictions['exp_total']:.2f}")
            with col3:
                st.metric(away_team, f"{predictions['exp_away']:.2f}")
            
            # Both Teams to Score
            st.subheader("🎯 Both Teams to Score")
            btts = calculate_btts(predictions)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Yes", f"{btts['yes']}%")
            with col2:
                st.metric("No", f"{btts['no']}%")
            
            # Asian Handicap
            st.subheader("📊 Asian Handicap")
            handicaps = calculate_asian_handicap(predictions)
            
            # Filter to show main handicap lines
            main_handicaps = [h for h in handicaps if h['line'] % 0.5 == 0]
            
            hcp_df = pd.DataFrame(main_handicaps)
            hcp_df['Handicap'] = hcp_df['line'].apply(lambda x: f"{x:+.1f}" if x != 0 else "0.0")
            hcp_df = hcp_df[['Handicap', 'home', 'away']]
            hcp_df.columns = ['Handicap', f'{home_team} %', f'{away_team} %']
            
            st.dataframe(hcp_df, use_container_width=True, hide_index=True, height=400)
            
            # Over/Under Goals
            st.subheader("🎯 Over/Under Goals")
            goal_lines = calculate_goal_lines(predictions)
            
            gl_df = pd.DataFrame(goal_lines)
            gl_df['Goals'] = gl_df['line'].apply(lambda x: f"{x:.1f}")
            gl_df = gl_df[['Goals', 'over', 'under']]
            gl_df.columns = ['Line', 'Over %', 'Under %']
            
            st.dataframe(gl_df, use_container_width=True, hide_index=True, height=350)
            
            # Top Scorelines
            st.subheader("📈 Most Likely Scores")
            top_scores = sorted(predictions['scoreline_probs'].items(), key=lambda x: x[1], reverse=True)[:10]
            
            score_data = []
            for (h, a), prob in top_scores:
                score_data.append({
                    'Score': f"{h}-{a}",
                    'Probability': f"{prob*100:.2f}%"
                })
            
            st.dataframe(pd.DataFrame(score_data), use_container_width=True, hide_index=True)
        
        else:
            st.error("❌ Cannot predict - insufficient data for selected teams")

else:
    st.info("👈 Select leagues and click 'Load Data' to begin")
    
    st.markdown("""
    ### 🌟 Features:
    - **Poisson Distribution Model** - Statistical match predictions
    - **Asian Handicap** - Full and half-goal lines
    - **Over/Under Goals** - Probability for each line
    - **BTTS** - Both Teams to Score predictions
    - **Most Likely Scores** - Top 10 scoreline probabilities
    
    ### 📊 Leagues:
    🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League | 🇪🇸 La Liga | 🇮🇹 Serie A | 🇮🇹 Serie B  
    🇩🇪 2.Bundesliga | 🇨🇭 Swiss Super League | 🇯🇵 J-League
    
    ### 📅 Seasons: 2023-24, 2024-25, 2025-26
    """)

st.markdown("---")
st.markdown("*Data from [football-data.co.uk](https://www.football-data.co.uk/) | Predictions for informational purposes only*")
