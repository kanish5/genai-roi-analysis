# Data Directory

## Structure

```
data/
├── raw/                    # Original survey data (not included)
│   └── survey_results_public.csv
│
└── processed/              # Cleaned datasets
    ├── genai_roi_clean.csv     # Analysis-ready data
    └── data_dictionary.csv     # Column definitions
```

## Getting the Raw Data

1. Visit: https://survey.stackoverflow.co/
2. Download the Stack Overflow Developer Survey 2024
3. Place `survey_results_public.csv` in the `data/raw/` folder
4. Run the cleaning pipeline: `python scripts/main.py`

## Processed Data

The cleaned dataset (`genai_roi_clean.csv`) contains:
- **60,907 rows** (survey respondents)
- **67 columns** (original + derived features)

Key derived columns:
- `AI_Status` - Using AI / Planning to Use / Not Using
- `Is_AI_User` - Binary flag (0/1)
- `Sentiment_Score` - 1-5 scale
- `Trust_Score` - 1-5 scale
- `ROI_Low/Mid/High` - Calculated ROI percentages
- `Company_Size_Category` - Standardized size buckets
- `Role_Category` - Simplified job roles
- `Region` - Geographic regions

See `data_dictionary.csv` for complete column documentation.
