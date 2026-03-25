"""
GenAI ROI Analysis - Analysis Module
=====================================

This module contains all analysis functions for generating
insights from the cleaned GenAI survey data.

Author: [Your Name]
Date: March 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


# =============================================================================
# ADOPTION ANALYSIS
# =============================================================================

def get_adoption_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get overall AI adoption summary statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics of AI adoption
    """
    summary = df['AI_Status'].value_counts().to_frame('Count')
    summary['Percentage'] = (summary['Count'] / len(df) * 100).round(1)
    return summary


def get_adoption_by_segment(
    df: pd.DataFrame, 
    segment_col: str
) -> pd.DataFrame:
    """
    Calculate AI adoption rate by a given segment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    segment_col : str
        Column name to segment by
        
    Returns:
    --------
    pd.DataFrame
        Adoption statistics by segment
    """
    adoption = df.groupby(segment_col).agg(
        Total=('Is_AI_User', 'count'),
        AI_Users=('Is_AI_User', 'sum')
    )
    adoption['Adoption_Rate'] = (adoption['AI_Users'] / adoption['Total'] * 100).round(1)
    adoption = adoption.sort_values('Adoption_Rate', ascending=False)
    return adoption


def get_adoption_by_company_size(df: pd.DataFrame) -> pd.DataFrame:
    """Get adoption rates by company size."""
    adoption = get_adoption_by_segment(df, 'Company_Size_Category')
    # Add sort order
    size_order = {
        'Solo/Freelancer': 1, 'Micro (2-9)': 2, 'Small (10-19)': 3,
        'Small-Mid (20-99)': 4, 'Mid (100-499)': 5, 'Mid-Large (500-999)': 6,
        'Large (1K-5K)': 7, 'Large (5K-10K)': 8, 'Enterprise (10K+)': 9
    }
    adoption['Sort_Order'] = adoption.index.map(size_order)
    return adoption.sort_values('Sort_Order')


def get_adoption_by_role(df: pd.DataFrame) -> pd.DataFrame:
    """Get adoption rates by job role."""
    return get_adoption_by_segment(df, 'Role_Category')


def get_adoption_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """Get adoption rates by geographic region."""
    return get_adoption_by_segment(df, 'Region')


def get_adoption_by_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Get adoption rates by experience level."""
    adoption = get_adoption_by_segment(df, 'Experience_Level')
    exp_order = {'Junior (0-2)': 1, 'Mid (2-5)': 2, 'Senior (5-10)': 3, 
                 'Expert (10+)': 4, 'Unknown': 5}
    adoption['Sort_Order'] = adoption.index.map(exp_order)
    return adoption.sort_values('Sort_Order')


# =============================================================================
# SENTIMENT ANALYSIS
# =============================================================================

def get_sentiment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of AI sentiment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Sentiment distribution statistics
    """
    sentiment = df['Sentiment_Category'].value_counts().to_frame('Count')
    sentiment['Percentage'] = (sentiment['Count'] / sentiment['Count'].sum() * 100).round(1)
    return sentiment


def get_sentiment_by_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get sentiment breakdown by AI adoption status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Cross-tabulation of sentiment by status
    """
    crosstab = pd.crosstab(
        df['AI_Status'], 
        df['Sentiment_Category'], 
        normalize='index'
    ) * 100
    return crosstab.round(1)


def get_average_sentiment_by_segment(
    df: pd.DataFrame, 
    segment_col: str
) -> pd.DataFrame:
    """
    Get average sentiment score by segment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    segment_col : str
        Column to segment by
        
    Returns:
    --------
    pd.DataFrame
        Average sentiment by segment
    """
    return df.groupby(segment_col)['Sentiment_Score'].agg(
        ['mean', 'median', 'std', 'count']
    ).round(2).sort_values('mean', ascending=False)


# =============================================================================
# BENEFITS & CHALLENGES ANALYSIS
# =============================================================================

def get_benefits_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of AI benefits selected by users.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Benefits summary with counts and percentages
    """
    # Filter to AI users
    ai_users = df[df['Is_AI_User'] == 1]
    
    benefit_cols = [col for col in df.columns if col.startswith('Ben_')]
    benefits = {}
    
    for col in benefit_cols:
        name = col.replace('Ben_', '').replace('_', ' ')
        benefits[name] = ai_users[col].sum()
    
    benefits_df = pd.DataFrame.from_dict(benefits, orient='index', columns=['Count'])
    benefits_df['Percentage'] = (benefits_df['Count'] / len(ai_users) * 100).round(1)
    return benefits_df.sort_values('Count', ascending=False)


def get_challenges_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary of AI challenges faced by users.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Challenges summary with counts and percentages
    """
    ai_users = df[df['Is_AI_User'] == 1]
    
    challenge_cols = [col for col in df.columns if col.startswith('Challenge_')]
    challenges = {}
    
    challenge_names = {
        'Challenge_Trust': 'Trust Issues',
        'Challenge_Context': 'Lacks Codebase Context',
        'Challenge_Security': 'Security Concerns',
        'Challenge_Training': 'Training Gaps',
        'Challenge_Adoption': 'Uneven Adoption'
    }
    
    for col in challenge_cols:
        name = challenge_names.get(col, col)
        challenges[name] = ai_users[col].sum()
    
    challenges_df = pd.DataFrame.from_dict(challenges, orient='index', columns=['Count'])
    challenges_df['Percentage'] = (challenges_df['Count'] / len(ai_users) * 100).round(1)
    return challenges_df.sort_values('Count', ascending=False)


# =============================================================================
# ROI ANALYSIS
# =============================================================================

def get_roi_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get ROI summary statistics across scenarios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        ROI summary by scenario
    """
    roi_stats = {
        'Conservative (2 hrs/wk)': {
            'Median_ROI': df['ROI_Low'].median(),
            'Mean_ROI': df['ROI_Low'].mean(),
            'Responses': df['ROI_Low'].notna().sum()
        },
        'Moderate (5 hrs/wk)': {
            'Median_ROI': df['ROI_Mid'].median(),
            'Mean_ROI': df['ROI_Mid'].mean(),
            'Responses': df['ROI_Mid'].notna().sum()
        },
        'Optimistic (8 hrs/wk)': {
            'Median_ROI': df['ROI_High'].median(),
            'Mean_ROI': df['ROI_High'].mean(),
            'Responses': df['ROI_High'].notna().sum()
        }
    }
    return pd.DataFrame(roi_stats).T.round(1)


def get_roi_by_segment(
    df: pd.DataFrame, 
    segment_col: str,
    scenario: str = 'Mid'
) -> pd.DataFrame:
    """
    Get ROI statistics by segment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    segment_col : str
        Column to segment by
    scenario : str
        ROI scenario ('Low', 'Mid', 'High')
        
    Returns:
    --------
    pd.DataFrame
        ROI statistics by segment
    """
    roi_col = f'ROI_{scenario}'
    roi_by_segment = df.groupby(segment_col)[roi_col].agg(
        ['median', 'mean', 'std', 'count']
    ).round(1).sort_values('median', ascending=False)
    return roi_by_segment


def get_breakeven_analysis(df: pd.DataFrame) -> Dict:
    """
    Calculate breakeven points for AI tool investment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    Dict
        Breakeven analysis results
    """
    median_hourly = df['Hourly_Rate_Est'].median()
    tool_cost_annual = 240  # $20/month
    
    # Hours needed to break even
    hours_breakeven = tool_cost_annual / median_hourly if median_hourly else None
    weeks_breakeven = hours_breakeven / 50 if hours_breakeven else None
    
    return {
        'median_hourly_rate': round(median_hourly, 2) if median_hourly else None,
        'tool_cost_annual': tool_cost_annual,
        'hours_to_breakeven_annual': round(hours_breakeven, 1) if hours_breakeven else None,
        'hours_per_week_to_breakeven': round(weeks_breakeven, 2) if weeks_breakeven else None
    }


# =============================================================================
# THREAT PERCEPTION ANALYSIS
# =============================================================================

def get_threat_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get distribution of job threat perception.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Threat perception distribution
    """
    threat = df['Threat_Category'].value_counts().to_frame('Count')
    threat['Percentage'] = (threat['Count'] / threat['Count'].sum() * 100).round(1)
    return threat


def get_threat_by_segment(
    df: pd.DataFrame, 
    segment_col: str
) -> pd.DataFrame:
    """
    Get threat perception by segment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    segment_col : str
        Column to segment by
        
    Returns:
    --------
    pd.DataFrame
        Cross-tabulation of threat by segment
    """
    crosstab = pd.crosstab(
        df[segment_col], 
        df['Threat_Category'], 
        normalize='index'
    ) * 100
    return crosstab.round(1)


# =============================================================================
# PRODUCTIVITY ANALYSIS
# =============================================================================

def get_time_searching_by_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare time spent searching by AI adoption status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    pd.DataFrame
        Time searching statistics by status
    """
    return df.groupby('AI_Status')['Time_Searching_Mins'].agg(
        ['mean', 'median', 'std', 'count']
    ).round(1)


# =============================================================================
# COMPREHENSIVE ANALYSIS REPORT
# =============================================================================

def generate_analysis_report(df: pd.DataFrame) -> Dict:
    """
    Generate a comprehensive analysis report.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
        
    Returns:
    --------
    Dict
        Complete analysis report with all insights
    """
    print("Generating comprehensive analysis report...")
    
    report = {
        'overview': {
            'total_responses': len(df),
            'ai_users': (df['Is_AI_User'] == 1).sum(),
            'adoption_rate': round((df['Is_AI_User'] == 1).mean() * 100, 1)
        },
        'adoption': {
            'summary': get_adoption_summary(df).to_dict(),
            'by_company_size': get_adoption_by_company_size(df).to_dict(),
            'by_role': get_adoption_by_role(df).to_dict(),
            'by_region': get_adoption_by_region(df).to_dict(),
            'by_experience': get_adoption_by_experience(df).to_dict()
        },
        'sentiment': {
            'distribution': get_sentiment_distribution(df).to_dict(),
            'by_status': get_sentiment_by_status(df).to_dict()
        },
        'benefits': get_benefits_summary(df).to_dict(),
        'challenges': get_challenges_summary(df).to_dict(),
        'roi': {
            'summary': get_roi_summary(df).to_dict(),
            'breakeven': get_breakeven_analysis(df)
        },
        'threat_perception': {
            'distribution': get_threat_distribution(df).to_dict()
        }
    }
    
    print("✅ Report generated successfully!")
    return report


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/processed/genai_roi_clean.csv")
    
    print("=" * 60)
    print("GenAI ROI Analysis - Summary Report")
    print("=" * 60)
    
    print("\n📊 ADOPTION SUMMARY")
    print(get_adoption_summary(df))
    
    print("\n🏢 ADOPTION BY COMPANY SIZE")
    print(get_adoption_by_company_size(df))
    
    print("\n👨‍💻 ADOPTION BY ROLE")
    print(get_adoption_by_role(df))
    
    print("\n💭 SENTIMENT DISTRIBUTION")
    print(get_sentiment_distribution(df))
    
    print("\n✅ BENEFITS SUMMARY")
    print(get_benefits_summary(df))
    
    print("\n⚠️ CHALLENGES SUMMARY")
    print(get_challenges_summary(df))
    
    print("\n💰 ROI SUMMARY")
    print(get_roi_summary(df))
    
    print("\n📉 BREAKEVEN ANALYSIS")
    print(get_breakeven_analysis(df))
