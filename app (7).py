import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import poisson
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
        
        # Check if we have the minimum required columns
        required_cols = ['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        if all(col in df.columns for col in required_cols):
            return df
        else:
            st.warning(f"Missing required columns in {league_code} season {season}")
            return None
    except Exception as e:
        st.warning(f"Could not load {league_code} season {season}: {str(e)}")
        return None

@st.cache_data
def load_all_leagues_data(selected_leagues):
    """Load data for selected leagues and seasons"""
    all_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    info_text = st.empty()
    
    total_items = sum(len(LEAGUE_MAPPINGS[league]['seasons']) for league in selected_leagues)
    current_item = 0
    loaded_count = 0
    
    for league_name in selected_leagues:
        league_info = LEAGUE_MAPPINGS[league_name]
        for season in league_info['seasons']:
            season_display = f"20{season[:2]}-20{season[2:]}"
            status_text.text(f"Loading {league_name} - {season_display}...")
            
            df = load_data_from_url(league_info['code'], season)
            if df is not None and not df.empty:
                # Only keep rows with valid data
                df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])
                if len(df) > 0:
                    df['League'] = league_name
                    df['Season'] = season
                    all_data.append(df)
                    loaded_count += 1
                    info_text.success(f"✅ Loaded {len(df)} matches from {league_name} {season_display}")
            
            current_item += 1
            progress_bar.progress(current_item / total_items)
    
    progress_bar.empty()
    status_text.empty()
    info_text.empty()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        st.success(f"✅ Successfully loaded {loaded_count} datasets with {len(combined_df)} total matches")
        return combined_df
    
    st.error("❌ Failed to load any data")
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

def predict_match_poisson(home_team, away_team, team_stats, df):
    """Predict match outcome using Poisson distribution"""
    
    if home_team not in team_stats or away_team not in team_stats:
        return None
    
    home_stats = team_stats[home_team]
    away_stats = team_stats[away_team]
    
    # Calculate league averages for context
    all_home_goals = df[df['HomeTeam'].notna()]['FTHG'].mean()
    all_away_goals = df[df['AwayTeam'].notna()]['FTAG'].mean()
    
    # Calculate attack and defense strengths
    home_attack_strength = home_stats['home_goals_scored'] / all_home_goals if all_home_goals > 0 else 1
    home_defense_strength = home_stats['home_goals_conceded'] / all_away_goals if all_away_goals > 0 else 1
    
    away_attack_strength = away_stats['away_goals_scored'] / all_away_goals if all_away_goals > 0 else 1
    away_defense_strength = away_stats['away_goals_conceded'] / all_home_goals if all_home_goals > 0 else 1
    
    # Expected goals using Poisson model
    expected_home_goals = home_attack_strength * away_defense_strength * all_home_goals
    expected_away_goals = away_attack_strength * home_defense_strength * all_away_goals
    
    # Ensure reasonable bounds
    expected_home_goals = max(0.1, min(5.0, expected_home_goals))
    expected_away_goals = max(0.1, min(5.0, expected_away_goals))
    
    # Calculate match outcome probabilities using Poisson distribution
    max_goals = 10  # Maximum goals to consider in calculations
    
    home_win_prob = 0
    draw_prob = 0
    away_win_prob = 0
    
    # Calculate probabilities for all possible scorelines
    for home_goals in range(max_goals + 1):
        for away_goals in range(max_goals + 1):
            # Probability of this exact scoreline
            prob = (poisson.pmf(home_goals, expected_home_goals) * 
                   poisson.pmf(away_goals, expected_away_goals))
            
            if home_goals > away_goals:
                home_win_prob += prob
            elif home_goals == away_goals:
                draw_prob += prob
            else:
                away_win_prob += prob
    
    # Normalize to ensure probabilities sum to 1
    total_prob = home_win_prob + draw_prob + away_win_prob
    if total_prob > 0:
        home_win_prob /= total_prob
        draw_prob /= total_prob
        away_win_prob /= total_prob
    
    # Calculate most likely scoreline
    most_likely_home = int(np.round(expected_home_goals))
    most_likely_away = int(np.round(expected_away_goals))
    
    # Calculate goal distribution probabilities for over/under
    goal_probabilities = {}
    for total_goals in range(max_goals + 1):
        prob = 0
        for home_goals in range(total_goals + 1):
            away_goals = total_goals - home_goals
            prob += (poisson.pmf(home_goals, expected_home_goals) * 
                    poisson.pmf(away_goals, expected_away_goals))
        goal_probabilities[total_goals] = prob
    
    return {
        'home_win_prob': home_win_prob,
        'draw_prob': draw_prob,
        'away_win_prob': away_win_prob,
        'expected_home_goals': expected_home_goals,
        'expected_away_goals': expected_away_goals,
        'expected_total_goals': expected_home_goals + expected_away_goals,
        'most_likely_score': f"{most_likely_home}-{most_likely_away}",
        'goal_probabilities': goal_probabilities
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
    """Calculate Over/Under goal lines using Poisson distribution"""
    goal_probs = predictions.get('goal_probabilities', {})
    
    lines = []
    
    for line in np.arange(0.5, 5.5, 0.5):
        over_prob = 0
        under_prob = 0
        
        # For half-goal lines
        threshold = int(np.floor(line))
        
        # Sum probabilities for over
        for goals in goal_probs.keys():
            if goals > threshold:
                over_prob += goal_probs[goals]
            elif goals <= threshold:
                under_prob += goal_probs[goals]
        
        # Normalize
        total = over_prob + under_prob
        if total > 0:
            over_prob = over_prob / total
            under_prob = under_prob / total
        
        lines.append({
            'line': line,
            'over_prob': round(over_prob * 100, 1),
            'under_prob': round(under_prob * 100, 1)
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
        predictions = predict_match_poisson(home_team, away_team, team_stats, df)
        
        if predictions:
            st.markdown("---")
            st.header("📈 Match Predictions")
            
            # Most Likely Score
            st.subheader("🎯 Most Likely Score")
            st.markdown(f"### {predictions['most_likely_score']}")
            
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
