# ⚽ Football Prediction App

A Streamlit-based football prediction application providing win probabilities, expected goals, Asian Handicap lines, and Over/Under predictions.

## 🚀 Quick Start

### Running on Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file path to: `app.py`
6. Click "Deploy"

### Running Locally

```bash
# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📊 Features

### Predictions
- **Win Probabilities**: Statistical likelihood of Home Win, Draw, or Away Win
- **Expected Goals**: Team-specific and total match goal predictions
- **Asian Handicap**: Coverage probabilities for lines from -3.0 to +3.0
- **Goal Lines**: Over/Under probabilities from 0.5 to 5.0 goals

### Data Coverage
- **Seasons**: 2023-2024, 2024-2025, 2025-2026
- **Leagues**:
  - 🏴󠁧󠁢󠁥󠁮󠁧󠁿 Premier League (England)
  - 🇪🇸 La Liga (Spain)
  - 🇮🇹 Serie A (Italy)
  - 🇮🇹 Serie B (Italy)
  - 🇩🇪 2. Bundesliga (Germany)
  - 🇨🇭 Swiss Super League (Switzerland)
  - 🇯🇵 J-League (Japan)

## 🎯 How to Use

1. **Select Leagues**: Choose which leagues to include in the analysis
2. **Load Data**: Click "Load Data" to fetch and process match data
3. **Select Teams**: Choose home and away teams from current season
4. **Get Predictions**: Click "Predict Match" to see comprehensive analysis

## 📈 Methodology

### Statistical Approach
- Calculates team performance metrics from historical data
- Uses goal-based analysis for expected outcomes
- Incorporates home/away performance differentials
- Applies probability normalization for accurate predictions

### Asian Handicap Calculation
- Based on expected goal differentials
- Adjusts probabilities according to handicap lines
- Provides coverage percentages for both teams

### Goal Lines
- Uses expected total goals as baseline
- Calculates probability distributions
- Covers standard betting lines from 0.5 to 5.0

## 📁 File Structure

```
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Requirements

- Python 3.8+
- streamlit
- pandas
- numpy

## 📊 Data Source

All match data is sourced from [football-data.co.uk](https://www.football-data.co.uk/)

### League Codes Used
- `E0` - Premier League
- `SP1` - La Liga
- `I1` - Serie A
- `I2` - Serie B
- `D2` - 2. Bundesliga
- `SWZ` - Swiss Super League
- `J1` - J-League

## ⚠️ Important Notes

- Requires internet connection to fetch data
- Data is cached for 1 hour to improve performance
- Predictions are for informational purposes only
- Some leagues may have limited 2025-2026 data early in season

## 🐛 Troubleshooting

### Data Not Loading
- Check internet connection
- Verify football-data.co.uk is accessible
- Try selecting fewer leagues
- Clear browser cache and reload

### No Teams Showing
- Ensure data loaded successfully
- Check that selected leagues have 2025-2026 season data
- Try using "All Leagues Combined" for statistics

### Prediction Errors
- Verify both teams have sufficient historical data
- Try different team combinations
- Reload data if issues persist

## 📝 License

This project is for educational and personal use only. 

## 🙏 Acknowledgments

- Data provided by football-data.co.uk
- Built with Streamlit

---

**Disclaimer**: Predictions are based on historical statistics and mathematical models. Use for informational purposes only.
