"""
GenAI ROI Analysis - Data Cleaning Module
==========================================

This module contains all data cleaning and transformation functions
for the GenAI ROI analysis project.

Author: [Your Name]
Date: March 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

# Columns to keep from the original survey
COLUMNS_TO_KEEP = [
    # AI/GenAI columns (CORE)
    'AISelect', 'AISent', 'AIBen', 'AIAcc', 'AIComplex', 
    'AITool', 'AINext', 'AIThreat', 'AIEthics', 'AIChallenges',
    'AISearchDev',
    
    # Segmentation
    'MainBranch', 'DevType', 'OrgSize', 'YearsCode', 'YearsCodePro',
    'Country', 'CompTotal', 'Currency', 'Industry', 'Employment',
    'RemoteWork', 'EdLevel', 'Age',
    
    # Productivity & Satisfaction
    'TimeSearching', 'TimeAnswering', 'JobSat',
    'PurchaseInfluence', 'ICorPM', 'WorkExp'
]

# ROI Calculation Assumptions
TOOL_COST_LOW = 120      # $10/month basic
TOOL_COST_MID = 240      # $20/month (Copilot)
TOOL_COST_HIGH = 480     # $40/month enterprise
HOURS_SAVED_LOW = 2      # Conservative: 2 hrs/week
HOURS_SAVED_MID = 5      # Moderate: 5 hrs/week
HOURS_SAVED_HIGH = 8     # Optimistic: 8 hrs/week
WORK_WEEKS_PER_YEAR = 50
WORK_HOURS_PER_YEAR = 2000


# =============================================================================
# MAPPING DICTIONARIES
# =============================================================================

SENTIMENT_MAP = {
    'Very unfavorable': 1,
    'Unfavorable': 2,
    'Indifferent': 3,
    'Unsure': 3,
    'Favorable': 4,
    'Very favorable': 5
}

SENTIMENT_CATEGORY_MAP = {
    'Very unfavorable': 'Negative',
    'Unfavorable': 'Negative',
    'Indifferent': 'Neutral',
    'Unsure': 'Neutral',
    'Favorable': 'Positive',
    'Very favorable': 'Positive'
}

TRUST_MAP = {
    'Highly distrust': 1,
    'Somewhat distrust': 2,
    'Neither trust nor distrust': 3,
    'Somewhat trust': 4,
    'Highly trust': 5
}

COMPLEXITY_MAP = {
    'Very poor at handling complex tasks': 1,
    'Bad at handling complex tasks': 2,
    'Neither good or bad at handling complex tasks': 3,
    'Good, but not great at handling complex tasks': 4,
    'Very well at handling complex tasks': 5
}

COMPANY_SIZE_MAP = {
    'Just me - I am a freelancer, sole proprietor, etc.': 'Solo/Freelancer',
    '2 to 9 employees': 'Micro (2-9)',
    '10 to 19 employees': 'Small (10-19)',
    '20 to 99 employees': 'Small-Mid (20-99)',
    '100 to 499 employees': 'Mid (100-499)',
    '500 to 999 employees': 'Mid-Large (500-999)',
    '1,000 to 4,999 employees': 'Large (1K-5K)',
    '5,000 to 9,999 employees': 'Large (5K-10K)',
    '10,000 or more employees': 'Enterprise (10K+)',
    "I don't know": 'Unknown'
}

COMPANY_SIZE_ORDER = {
    'Solo/Freelancer': 1, 'Micro (2-9)': 2, 'Small (10-19)': 3,
    'Small-Mid (20-99)': 4, 'Mid (100-499)': 5, 'Mid-Large (500-999)': 6,
    'Large (1K-5K)': 7, 'Large (5K-10K)': 8, 'Enterprise (10K+)': 9, 
    'Unknown': 0
}

THREAT_MAP = {
    'No': 'Not Threatened',
    'Yes': 'Threatened',
    "I'm not sure": 'Uncertain'
}

TIME_SEARCHING_MAP = {
    'Less than 15 minutes a day': 7.5,
    '15-30 minutes a day': 22.5,
    '30-60 minutes a day': 45,
    '60-120 minutes a day': 90,
    'Over 120 minutes a day': 150
}

# Benefits to extract
BENEFITS_LIST = [
    'Increase productivity',
    'Speed up learning', 
    'Greater efficiency',
    'Improve accuracy in coding',
    'Make workload more manageable',
    'Improve collaboration'
]


# =============================================================================
# CLEANING FUNCTIONS
# =============================================================================

def load_and_filter_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw survey data and filter to relevant columns.
    
    Parameters:
    -----------
    filepath : str
        Path to the survey_results_public.csv file
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe with only relevant columns
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    # Keep only columns that exist
    cols_to_keep = [col for col in COLUMNS_TO_KEEP if col in df.columns]
    df_filtered = df[cols_to_keep]
    
    # Filter to respondents who answered AI questions
    df_filtered = df_filtered[df_filtered['AISelect'].notna()]
    
    print(f"Filtered shape: {df_filtered.shape}")
    return df_filtered


def clean_ai_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cleaned AI adoption status columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AISelect column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with added AI_Status and Is_AI_User columns
    """
    df = df.copy()
    
    # Simplified status
    df['AI_Status'] = df['AISelect'].map({
        'Yes': 'Using AI',
        'No, but I plan to soon': 'Planning to Use',
        "No, and I don't plan to": 'Not Using'
    })
    
    # Binary flag
    df['Is_AI_User'] = (df['AISelect'] == 'Yes').astype(int)
    
    return df


def clean_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sentiment score and category columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AISent column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Sentiment_Score and Sentiment_Category columns
    """
    df = df.copy()
    df['Sentiment_Score'] = df['AISent'].map(SENTIMENT_MAP)
    df['Sentiment_Category'] = df['AISent'].map(SENTIMENT_CATEGORY_MAP)
    return df


def clean_trust_and_complexity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trust and complexity score columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AIAcc and AIComplex columns
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Trust_Score and Complexity_Score columns
    """
    df = df.copy()
    df['Trust_Score'] = df['AIAcc'].map(TRUST_MAP)
    df['Complexity_Score'] = df['AIComplex'].map(COMPLEXITY_MAP)
    return df


def clean_company_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create standardized company size categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with OrgSize column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Company_Size_Category and Company_Size_Order columns
    """
    df = df.copy()
    df['Company_Size_Category'] = df['OrgSize'].map(COMPANY_SIZE_MAP)
    df['Company_Size_Order'] = df['Company_Size_Category'].map(COMPANY_SIZE_ORDER)
    return df


def clean_years_experience(val) -> float:
    """
    Convert years of experience to numeric value.
    
    Parameters:
    -----------
    val : str or numeric
        Raw years value from survey
        
    Returns:
    --------
    float
        Numeric years value
    """
    if pd.isna(val):
        return np.nan
    if val == 'Less than 1 year':
        return 0.5
    if val == 'More than 50 years':
        return 50
    try:
        return float(val)
    except:
        return np.nan


def get_experience_level(years: float) -> str:
    """
    Categorize years of experience into levels.
    
    Parameters:
    -----------
    years : float
        Numeric years of experience
        
    Returns:
    --------
    str
        Experience level category
    """
    if pd.isna(years):
        return 'Unknown'
    if years < 2:
        return 'Junior (0-2)'
    elif years < 5:
        return 'Mid (2-5)'
    elif years < 10:
        return 'Senior (5-10)'
    else:
        return 'Expert (10+)'


def clean_experience(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cleaned experience columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with YearsCodePro column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Years_Pro_Clean and Experience_Level columns
    """
    df = df.copy()
    df['Years_Pro_Clean'] = df['YearsCodePro'].apply(clean_years_experience)
    df['Experience_Level'] = df['Years_Pro_Clean'].apply(get_experience_level)
    return df


def simplify_role(role: str) -> str:
    """
    Simplify job role into standard categories.
    
    Parameters:
    -----------
    role : str
        Original role from survey
        
    Returns:
    --------
    str
        Simplified role category
    """
    if pd.isna(role):
        return 'Unknown'
    role = str(role).lower()
    
    if 'full-stack' in role:
        return 'Full-Stack Developer'
    elif 'back-end' in role:
        return 'Back-End Developer'
    elif 'front-end' in role:
        return 'Front-End Developer'
    elif 'mobile' in role:
        return 'Mobile Developer'
    elif 'data scientist' in role or 'machine learning' in role:
        return 'Data Scientist/ML'
    elif 'data engineer' in role:
        return 'Data Engineer'
    elif 'devops' in role or 'sre' in role:
        return 'DevOps/SRE'
    elif 'manager' in role or 'executive' in role:
        return 'Manager/Executive'
    elif 'student' in role:
        return 'Student'
    elif 'academic' in role or 'research' in role:
        return 'Researcher/Academic'
    else:
        return 'Other Developer'


def clean_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simplified role categories.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with DevType column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Role_Category column
    """
    df = df.copy()
    df['Role_Category'] = df['DevType'].apply(simplify_role)
    return df


def clean_threat_perception(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create threat perception category.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AIThreat column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Threat_Category column
    """
    df = df.copy()
    df['Threat_Category'] = df['AIThreat'].map(THREAT_MAP)
    return df


def extract_benefits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary columns for each AI benefit.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AIBen column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with benefit flag columns and Benefits_Count
    """
    df = df.copy()
    
    for benefit in BENEFITS_LIST:
        col_name = 'Ben_' + benefit.replace(' ', '_')[:20]
        df[col_name] = df['AIBen'].fillna('').str.contains(benefit, case=False).astype(int)
    
    df['Benefits_Count'] = df['AIBen'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    return df


def extract_challenges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract binary columns for each AI challenge.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with AIChallenges column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with challenge flag columns and Challenges_Count
    """
    df = df.copy()
    
    df['Challenge_Trust'] = df['AIChallenges'].fillna('').str.contains(
        'trust', case=False).astype(int)
    df['Challenge_Context'] = df['AIChallenges'].fillna('').str.contains(
        'context|codebase', case=False).astype(int)
    df['Challenge_Security'] = df['AIChallenges'].fillna('').str.contains(
        'security', case=False).astype(int)
    df['Challenge_Training'] = df['AIChallenges'].fillna('').str.contains(
        'training', case=False).astype(int)
    df['Challenge_Adoption'] = df['AIChallenges'].fillna('').str.contains(
        'Not everyone', case=False).astype(int)
    
    df['Challenges_Count'] = df['AIChallenges'].fillna('').apply(
        lambda x: len(x.split(';')) if x else 0
    )
    
    return df


def clean_time_searching(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert time searching to numeric minutes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with TimeSearching column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Time_Searching_Mins column
    """
    df = df.copy()
    df['Time_Searching_Mins'] = df['TimeSearching'].map(TIME_SEARCHING_MAP)
    return df


def clean_compensation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean compensation data and calculate hourly rate.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with CompTotal column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Comp_Clean and Hourly_Rate_Est columns
    """
    df = df.copy()
    
    # Copy original
    df['Comp_Clean'] = df['CompTotal']
    
    # Remove outliers
    df.loc[df['Comp_Clean'] > 1000000, 'Comp_Clean'] = np.nan
    df.loc[df['Comp_Clean'] < 1000, 'Comp_Clean'] = np.nan
    
    # Calculate hourly rate
    df['Hourly_Rate_Est'] = df['Comp_Clean'] / WORK_HOURS_PER_YEAR
    
    return df


def calculate_roi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ROI metrics for different scenarios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Hourly_Rate_Est column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with ROI calculation columns
    """
    df = df.copy()
    
    # Hours saved assumptions
    df['Hours_Saved_Weekly_Low'] = HOURS_SAVED_LOW
    df['Hours_Saved_Weekly_Mid'] = HOURS_SAVED_MID
    df['Hours_Saved_Weekly_High'] = HOURS_SAVED_HIGH
    
    # Calculate annual value
    df['Value_Annual_Low'] = (df['Hours_Saved_Weekly_Low'] * 
                              df['Hourly_Rate_Est'] * WORK_WEEKS_PER_YEAR)
    df['Value_Annual_Mid'] = (df['Hours_Saved_Weekly_Mid'] * 
                              df['Hourly_Rate_Est'] * WORK_WEEKS_PER_YEAR)
    df['Value_Annual_High'] = (df['Hours_Saved_Weekly_High'] * 
                               df['Hourly_Rate_Est'] * WORK_WEEKS_PER_YEAR)
    
    # Calculate ROI
    df['ROI_Low'] = ((df['Value_Annual_Low'] - TOOL_COST_MID) / 
                     TOOL_COST_MID * 100).round(1)
    df['ROI_Mid'] = ((df['Value_Annual_Mid'] - TOOL_COST_MID) / 
                     TOOL_COST_MID * 100).round(1)
    df['ROI_High'] = ((df['Value_Annual_High'] - TOOL_COST_MID) / 
                      TOOL_COST_MID * 100).round(1)
    
    return df


def calculate_perception_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate composite AI perception score.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Sentiment_Score, Trust_Score, Complexity_Score
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with AI_Perception_Score and AI_Perception_Category
    """
    df = df.copy()
    
    # Average of available scores
    df['AI_Perception_Score'] = (
        df['Sentiment_Score'].fillna(3) + 
        df['Trust_Score'].fillna(3) + 
        df['Complexity_Score'].fillna(3)
    ) / 3
    
    # Categorize
    df['AI_Perception_Category'] = pd.cut(
        df['AI_Perception_Score'], 
        bins=[0, 2, 3, 4, 5], 
        labels=['Skeptical', 'Neutral', 'Positive', 'Enthusiastic']
    )
    
    return df


def get_region(country: str) -> str:
    """
    Map country to geographic region.
    
    Parameters:
    -----------
    country : str
        Country name
        
    Returns:
    --------
    str
        Geographic region
    """
    if pd.isna(country):
        return 'Unknown'
    country = str(country)
    
    north_america = ['United States of America', 'Canada']
    western_europe = ['United Kingdom of Great Britain and Northern Ireland', 
                      'Germany', 'France', 'Netherlands', 'Sweden', 'Poland', 
                      'Italy', 'Spain', 'Switzerland', 'Austria', 'Belgium', 
                      'Denmark', 'Norway', 'Finland', 'Ireland', 'Portugal', 
                      'Czech Republic']
    eastern_europe = ['Ukraine', 'Russian Federation', 'Romania', 'Bulgaria', 
                      'Hungary', 'Serbia']
    south_asia = ['India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal']
    east_se_asia = ['China', 'Japan', 'South Korea', 'Taiwan', 'Hong Kong', 
                    'Singapore', 'Vietnam', 'Thailand', 'Indonesia', 'Malaysia', 
                    'Philippines']
    latin_america = ['Brazil', 'Argentina', 'Mexico', 'Colombia', 'Chile']
    oceania = ['Australia', 'New Zealand']
    middle_east = ['Israel', 'Turkey', 'Iran', 'United Arab Emirates', 
                   'Saudi Arabia', 'Egypt']
    africa = ['South Africa', 'Nigeria', 'Kenya']
    
    if country in north_america:
        return 'North America'
    elif country in western_europe:
        return 'Western Europe'
    elif country in eastern_europe:
        return 'Eastern Europe'
    elif country in south_asia:
        return 'South Asia'
    elif country in east_se_asia:
        return 'East/SE Asia'
    elif country in latin_america:
        return 'Latin America'
    elif country in oceania:
        return 'Oceania'
    elif country in middle_east:
        return 'Middle East'
    elif country in africa:
        return 'Africa'
    else:
        return 'Other'


def add_regions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add geographic region column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with Country column
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with Region column
    """
    df = df.copy()
    df['Region'] = df['Country'].apply(get_region)
    return df


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete data cleaning pipeline.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Fully cleaned and enriched dataframe
    """
    print("Starting data cleaning pipeline...")
    
    # Apply all cleaning functions
    df = clean_ai_status(df)
    print("  ✓ AI status cleaned")
    
    df = clean_sentiment(df)
    print("  ✓ Sentiment cleaned")
    
    df = clean_trust_and_complexity(df)
    print("  ✓ Trust and complexity cleaned")
    
    df = clean_company_size(df)
    print("  ✓ Company size cleaned")
    
    df = clean_experience(df)
    print("  ✓ Experience cleaned")
    
    df = clean_roles(df)
    print("  ✓ Roles cleaned")
    
    df = clean_threat_perception(df)
    print("  ✓ Threat perception cleaned")
    
    df = extract_benefits(df)
    print("  ✓ Benefits extracted")
    
    df = extract_challenges(df)
    print("  ✓ Challenges extracted")
    
    df = clean_time_searching(df)
    print("  ✓ Time searching cleaned")
    
    df = clean_compensation(df)
    print("  ✓ Compensation cleaned")
    
    df = calculate_roi(df)
    print("  ✓ ROI calculated")
    
    df = calculate_perception_score(df)
    print("  ✓ Perception score calculated")
    
    df = add_regions(df)
    print("  ✓ Regions added")
    
    print(f"\nCleaning complete! Final shape: {df.shape}")
    return df


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Default paths
    input_path = "data/raw/survey_results_public.csv"
    output_path = "data/processed/genai_roi_clean.csv"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    # Run pipeline
    df_raw = load_and_filter_data(input_path)
    df_clean = clean_survey_data(df_raw)
    
    # Save cleaned data
    df_clean.to_csv(output_path, index=False)
    print(f"\n✅ Cleaned data saved to: {output_path}")
