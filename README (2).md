# Football Prediction App ⚽

A Streamlit-based football prediction application that provides:
- Win probabilities (Home/Draw/Away)
- Expected goals predictions
- Asian Handicap lines with coverage probabilities
- Over/Under goal lines

## Features

### Data Sources
- Pulls data directly from [football-data.co.uk](https://www.football-data.co.uk/)
- Covers seasons: 2023-2024, 2024-2025, 2025-2026
- Includes 7 major leagues:
  - Premier League (England)
  - La Liga (Spain)
  - Serie A (Italy)
  - Serie B (Italy)
  - 2.Bundesliga (Germany)
  - Swiss Super League (Switzerland)
  - J-League (Japan)

### Predictions
1. **Win Probabilities**: Statistical likelihood of Home Win, Draw, or Away Win
2. **Expected Goals**: Team-specific and total match goal predictions
3. **Asian Handicap**: Coverage probabilities for handicap lines from -3.0 to +3.0
4. **Goal Lines**: Over/Under probabilities for lines from 0.5 to 5.0 goals

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run football_app.py
```

2. The app will open in your default web browser

3. In the sidebar:
   - Select the leagues you want to include
   - Click "Load Data & Train Model" to fetch data and train the prediction model

4. Once the model is trained:
   - Select home and away teams from the 2025-2026 season
   - Click "Predict Match" to see predictions

5. View results:
   - Win probabilities
   - Expected goals
   - Asian Handicap lines
   - Over/Under goal lines

## How It Works

### Data Processing
- Fetches CSV files from football-data.co.uk for each league and season
- Combines data from multiple seasons for robust predictions
- Calculates team statistics including:
  - Goals scored/conceded (home and away)
  - Win percentages
  - Historical performance metrics

### Machine Learning Models
- **Result Prediction**: Random Forest Classifier for match outcomes (H/D/A)
- **Goals Prediction**: Gradient Boosting Regressor for total goals

### Calculations
- **Asian Handicap**: Probabilities based on goal difference expectations
- **Goal Lines**: Probability distributions based on expected total goals
- **Team Stats**: Rolling averages and form indicators

## League Codes

The app uses the following league codes from football-data.co.uk:
- `E0` - Premier League
- `SP1` - La Liga
- `I1` - Serie A
- `I2` - Serie B
- `D2` - 2.Bundesliga
- `SWZ` - Swiss Super League
- `J1` - J-League

## Notes

- The app requires an internet connection to fetch data from football-data.co.uk
- Data is cached for 1 hour to improve performance
- Predictions are based on historical data and statistical models
- For best results, ensure all leagues have loaded successfully before making predictions

## Troubleshooting

### Data Loading Issues
- Check your internet connection
- Verify that football-data.co.uk is accessible
- Some seasons may not have data for all leagues yet

### Prediction Errors
- Ensure teams have sufficient historical data
- Try selecting different leagues
- Reload data if predictions seem inconsistent

## Future Enhancements

Potential improvements:
- Add more leagues
- Include additional betting markets
- Implement form-based adjustments
- Add head-to-head statistics
- Include player availability impact
- Export predictions to CSV

## Data Attribution

All match data is sourced from [football-data.co.uk](https://www.football-data.co.uk/).
Please visit their website for data usage terms and conditions.

## License

This project is for educational and personal use only.
