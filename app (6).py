import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
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
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')
        return df
    except Exception as e:
        return None

@st.cache_data
def load_all_leagues_data(selected_leagues):
    """Load data for selected leagues and seasons"""
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_items = sum(len(LEAGUE_MAPPINGS[league]['seasons']) for league in selected_leagues)
    current_item = 0
    
    for league_name in selected_leagues:
        league_info = LEAGUE_MAPPINGS[league_name]
        for season in league_info['seasons']:
            status_text.text(f"Loading {league_name} - Season {season}...")
            df = load_data_from_url(league_info['code'], season)
            if df is not None and not df.empty:
                df['League'] = league_name
                df['Season'] = season
                all_data.append(df)
            current_item += 1
            progress_bar.progress(current_item / total_items)
    
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    return pd.DataFrame()

def calculate_team_stats(df, league_filter=None):
    """Calculate comprehensive team statistics"""
    if league_filter:
        df = df[df['League'] == league_filter].copy()
    
    stats_dict = {}
    
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        if pd.isna(team):
            continue
            
        # Home games
        home_games = df[df['HomeTeam'] == team].copy()
        # Away games  
        away_games = df[df['AwayTeam'] == team].copy()
        
        # Calculate statistics
        total_games = len(home_games) + len(away_games)
        
        if total_games == 0:
            continue
        
        # Home stats
        home_goals_scored = home_games['FTHG'].mean() if len(home_games) > 0 else 0
        home_goals_conceded = home_games['FTAG'].mean() if len(home_games) > 0 else 0
        home_wins = (home_games['FTR'] == 'H').sum() if len(home_games) > 0 else 0
        home_draws = (home_games['FTR'] == 'D').sum() if len(home_games) > 0 else 0
        
        # Away stats
        away_goals_scored = away_games['FTAG'].mean() if len(away_games) > 0 else 0
        away_goals_conceded = away_games['FTHG'].mean() if len(away_games) > 0 else 0
        away_wins = (away_games['FTR'] == 'A').sum() if len(away_games) > 0 else 0
        away_draws = (away_games['FTR'] == 'D').sum() if len(away_games) > 0 else 0
        
        # Overall stats
        total_wins = home_wins + away_wins
        total_draws = home_draws + away_draws
        
        stats_dict[team] = {
            'games_played': total_games,
            'home_games': len(home_games),
            'away_games': len(away_games),
            'home_goals_scored': round(home_goals_scored, 2),
            'home_goals_conceded': round(home_goals_conceded, 2),
            'away_goals_scored': round(away_goals_scored, 2),
            'away_goals_conceded': round(away_goals_conceded, 2),
            'avg_goals_scored': round((home_goals_scored + away_goals_scored) / 2, 2),
            'avg_goals_conceded': round((home_goals_conceded + away_goals_conceded) / 2, 2),
            'win_rate': round((total_wins / total_games) * 100, 1) if total_games > 0 else 0,
            'draw_rate': round((total_draws / total_games) * 100, 1) if total_games > 0 else 0
        }
    
    return stats_dict

def predict_match(home_team, away_team, team_stats):
    """Predict match outcome using team statistics"""
    
    if home_team not in team_stats or away_team not in team_stats:
        return None
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Expected goals calculation
    expected_home_goals = (home_stats['home_goals_scored'] + away_stats['away_goals_conceded']) / 2
    expected_away_goals = (away_stats['away_goals_scored'] + home_stats['home_goals_conceded']) / 2
    
    # Goal difference
    goal_diff = expected_home_goals - expected_away_goals
    
    # Win probability calculation (simplified Poisson-based approach)
    if goal_diff > 0.5:
        home_win_prob = min(0.85, 0.45 + (goal_diff * 0.15))
        away_win_prob = max(0.05, 0.25 - (goal_diff * 0.10))
    elif goal_diff < -0.5:
        home_win_prob = max(0.05, 0.25 + (goal_diff * 0.10))
        away_win_prob = min(0.85, 0.45 - (goal_diff * 0.15))
    else:
        home_win_prob = 0.35
        away_win_prob = 0.30
    
    draw_prob = max(0.10, 1.0 - home_win_prob - away_win_prob)
    
    # Normalize probabilities
    total_prob = home_win_prob + draw_prob + away_win_prob
    home_win_prob /= total_prob
    draw_prob /= total_prob
    away_win_prob /= total_prob
    
    return {
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals,
        'expected_total_goals': expected_home_goals + expected_away_goals
    }

def calculate_asian_handicap(predictions):
    """Calculate Asian Handicap probabilities"""
    goal_diff = predictions['expected_home_goals'] - predictions['expected_away_goals']
    home_win_prob = predictions['home_win_prob']
    away_win_prob = predictions['away_win_prob']
    
    handicaps = []
    
    for line in np.arange(-3.0, 3.5, 0.5):
        # Adjust probabilities based on handicap line
        adjusted_diff = goal_diff - line
        
        if adjusted_diff > 0.3:
            home_cover = min(0.95, home_win_prob + (adjusted_diff * 0.10))
        elif adjusted_diff < -0.3:
            home_cover = max(0.05, home_win_prob - (abs(adjusted_diff) * 0.10))
        else:
            home_cover = home_win_prob
        
        handicaps.append({
            'line': line,
            'home_cover': round(home_cover * 100, 1),
            'away_cover': round((1 - home_cover) * 100, 1)
        })
    
    return handicaps

def calculate_goal_lines(predictions):
    """Calculate Over/Under goal lines"""
    expected_total = predictions['expected_total_goals']
    
    lines = []
    
    for line in np.arange(0.5, 5.5, 0.5):
        diff = expected_total - line
        
        # Probability calculation based on distance from expected
        if diff > 0.5:
            over_prob = min(0.95, 0.55 + (diff * 0.12))
        elif diff < -0.5:
            over_prob = max(0.05, 0.45 + (diff * 0.12))
        else:
            over_prob = 0.50
        
        lines.append({
            'line': line,
            'over_prob': round(over_prob * 100, 1),
            'under_prob': round((1 - over_prob) * 100, 1)
        })
    
    return lines

# Main App
st.title("⚽ Football Prediction App")
st.markdown("### Asian Handicap & Goal Lines Predictor")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    selected_leagues = st.multiselect(
        "Select Leagues",
        list(LEAGUE_MAPPINGS.keys()),
        default=['Premier League', 'La Liga', 'Serie A']
    )
    
    if st.button("🔄 Load Data", type="primary", use_container_width=True):
        if not selected_leagues:
            st.error("Please select at least one league")
        else:
            with st.spinner("Loading data..."):
                df = load_all_leagues_data(selected_leagues)
                
                if not df.empty:
                    # Filter for required columns
                    required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'League', 'Season']
                    available_cols = [col for col in required_cols if col in df.columns]
                    df = df[available_cols].dropna(subset=['FTHG', 'FTAG', 'FTR'])
                    
                    st.session_state.df = df
                    st.session_state.data_loaded = True
                    st.success(f"✅ Loaded {len(df)} matches from {len(selected_leagues)} leagues")
                    st.rerun()
                else:
                    st.error("Failed to load data. Please check your internet connection.")

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Main content
if st.session_state.data_loaded:
    df = st.session_state.df
    
    # Get current season teams
    current_season_df = df[df['Season'] == '2526']
    
    if current_season_df.empty:
        st.warning("⚠️ No data available for 2025-2026 season yet. Using latest available season.")
        current_season_df = df[df['Season'] == df['Season'].max()]
    
    all_teams = sorted(pd.concat([current_season_df['HomeTeam'], 
                                   current_season_df['AwayTeam']]).dropna().unique())
    
    # League filter for statistics
    st.sidebar.markdown("---")
    stat_league = st.sidebar.selectbox(
        "Calculate stats for:",
        ["All Leagues Combined"] + selected_leagues
    )
    
    # Calculate team statistics
    if stat_league == "All Leagues Combined":
        team_stats = calculate_team_stats(df)
    else:
        team_stats = calculate_team_stats(df, stat_league)
    
    # Match prediction interface
    st.header("🎯 Match Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox("🏠 Home Team", all_teams, key='home')
    
    with col2:
        away_teams_filtered = [t for t in all_teams if t != home_team]
        away_team = st.selectbox("✈️ Away Team", away_teams_filtered, key='away')
    
    if st.button("📊 Predict Match", type="primary", use_container_width=True):
        predictions = predict_match(home_team, away_team, team_stats)
        
        if predictions:
            st.markdown("---")
            st.header("📈 Match Predictions")
            
            # Win Probabilities
            st.subheader("🏆 Win Probabilities")
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                st.metric(
                    f"{home_team} Win", 
                    f"{predictions['home_win_prob']*100:.1f}%",
                    delta=None
                )
            with prob_col2:
                st.metric(
                    "Draw", 
                    f"{predictions['draw_prob']*100:.1f}%",
                    delta=None
                )
            with prob_col3:
                st.metric(
                    f"{away_team} Win", 
                    f"{predictions['away_win_prob']*100:.1f}%",
                    delta=None
                )
            
            # Expected Goals
            st.subheader("⚽ Expected Goals")
            goal_col1, goal_col2, goal_col3 = st.columns(3)
            
            with goal_col1:
                st.metric(f"{home_team}", f"{predictions['expected_home_goals']:.2f}")
            with goal_col2:
                st.metric("Total", f"{predictions['expected_total_goals']:.2f}")
            with goal_col3:
                st.metric(f"{away_team}", f"{predictions['expected_away_goals']:.2f}")
            
            # Asian Handicap
            st.subheader("📊 Asian Handicap Lines")
            handicaps = calculate_asian_handicap(predictions)
            
            handicap_df = pd.DataFrame(handicaps)
            handicap_df['Line'] = handicap_df['line'].apply(lambda x: f"{x:+.1f}")
            handicap_df = handicap_df[['Line', 'home_cover', 'away_cover']]
            handicap_df.columns = ['Handicap Line', f'{home_team} Cover %', f'{away_team} Cover %']
            
            st.dataframe(
                handicap_df,
                use_container_width=True,
                hide_index=True,
                height=400
            )
            
            # Goal Lines
            st.subheader("🎯 Over/Under Goal Lines")
            goal_lines = calculate_goal_lines(predictions)
            
            goal_lines_df = pd.DataFrame(goal_lines)
            goal_lines_df['Line'] = goal_lines_df['line'].apply(lambda x: f"{x:.1f}")
            goal_lines_df = goal_lines_df[['Line', 'over_prob', 'under_prob']]
            goal_lines_df.columns = ['Goal Line', 'Over %', 'Under %']
            
            st.dataframe(
                goal_lines_df,
                use_container_width=True,
                hide_index=True,
                height=350
            )
            
        else:
            st.error("❌ Unable to make predictions. Teams may not have enough historical data.")
    
    # Team Statistics
    with st.expander("📊 View Team Statistics", expanded=False):
        if team_stats:
            stats_df = pd.DataFrame(team_stats).T
            stats_df = stats_df.round(2)
            stats_df = stats_df.sort_values('win_rate', ascending=False)
            
            st.dataframe(
                stats_df,
                use_container_width=True,
                height=500
            )
        else:
            st.info("No statistics available")

else:
    # Welcome screen
    st.info("👈 Click 'Load Data' in the sidebar to get started")
    
    st.markdown("""
    ### 🌟 Features:
    
    - **Win Probabilities** - Home, Draw, Away predictions
    - **Expected Goals** - Team-specific and total match goals  
    - **Asian Handicap Lines** - Coverage probabilities from -3.0 to +3.0
    - **Over/Under Lines** - Goal line probabilities from 0.5 to 5.0
    
    ### 📊 Leagues Available:
    
    - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League
    - 🇪🇸 La Liga
    - 🇮🇹 Serie A
    - 🇮🇹 Serie B
    - 🇩🇪 2. Bundesliga
    - 🇨🇭 Swiss Super League
    - 🇯🇵 J-League
    
    ### 📅 Data Coverage:
    
    - Season 2023-2024
    - Season 2024-2025
    - Season 2025-2026 (Current)
    
    ### 🔗 Data Source:
    
    All data from [football-data.co.uk](https://www.football-data.co.uk/)
    """)

# Footer
st.markdown("---")
st.markdown("*Predictions are based on historical statistics and should be used for informational purposes only.*")
