"""
GenAI ROI Analysis - Visualization Module
==========================================

This module contains all visualization functions for creating
charts and plots for the GenAI ROI analysis project.

Author: [Your Name]
Date: March 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE CONFIGURATION
# =============================================================================

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Custom color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#28A745',
    'warning': '#FFC107',
    'danger': '#DC3545',
    'info': '#17A2B8',
    'light': '#F8F9FA',
    'dark': '#343A40'
}

# Palette for multiple categories
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#28A745', '#6C757D', 
           '#17A2B8', '#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

# Figure settings
FIG_SIZE_SMALL = (8, 5)
FIG_SIZE_MEDIUM = (10, 6)
FIG_SIZE_LARGE = (12, 8)
FIG_SIZE_WIDE = (14, 6)

def set_plot_style():
    """Set consistent plot styling."""
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

set_plot_style()


# =============================================================================
# ADOPTION VISUALIZATIONS
# =============================================================================

def plot_adoption_overview(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create pie chart of AI adoption status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    
    adoption = df['AI_Status'].value_counts()
    colors = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    
    wedges, texts, autotexts = ax.pie(
        adoption.values, 
        labels=adoption.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=(0.02, 0.02, 0.02),
        shadow=False,
        startangle=90
    )
    
    # Style
    plt.setp(autotexts, size=11, weight='bold', color='white')
    ax.set_title('AI Tool Adoption Status\n(n={:,})'.format(len(df)), 
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_adoption_by_company_size(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create bar chart of adoption rate by company size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    # Calculate adoption rate
    adoption = df.groupby('Company_Size_Category')['Is_AI_User'].mean() * 100
    
    # Sort by company size order
    size_order = ['Solo/Freelancer', 'Micro (2-9)', 'Small (10-19)', 
                  'Small-Mid (20-99)', 'Mid (100-499)', 'Mid-Large (500-999)',
                  'Large (1K-5K)', 'Large (5K-10K)', 'Enterprise (10K+)']
    adoption = adoption.reindex([s for s in size_order if s in adoption.index])
    
    # Create bar chart
    bars = ax.bar(range(len(adoption)), adoption.values, color=COLORS['primary'], 
                  edgecolor='white', linewidth=1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, adoption.values)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Style
    ax.set_xticks(range(len(adoption)))
    ax.set_xticklabels(adoption.index, rotation=45, ha='right')
    ax.set_ylabel('Adoption Rate (%)')
    ax.set_xlabel('Company Size')
    ax.set_title('AI Tool Adoption by Company Size', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.axhline(y=df['Is_AI_User'].mean()*100, color=COLORS['danger'], 
               linestyle='--', linewidth=2, label=f'Average ({df["Is_AI_User"].mean()*100:.1f}%)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_adoption_by_role(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create horizontal bar chart of adoption rate by role.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    # Calculate adoption rate
    adoption = df.groupby('Role_Category')['Is_AI_User'].mean() * 100
    adoption = adoption.sort_values(ascending=True)
    
    # Remove Unknown
    adoption = adoption.drop('Unknown', errors='ignore')
    
    # Create horizontal bar chart
    colors = [COLORS['primary'] if v >= 60 else COLORS['secondary'] for v in adoption.values]
    bars = ax.barh(range(len(adoption)), adoption.values, color=colors, edgecolor='white')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, adoption.values)):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=9)
    
    # Style
    ax.set_yticks(range(len(adoption)))
    ax.set_yticklabels(adoption.index)
    ax.set_xlabel('Adoption Rate (%)')
    ax.set_title('AI Tool Adoption by Job Role', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 85)
    ax.axvline(x=df['Is_AI_User'].mean()*100, color=COLORS['danger'], 
               linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# SENTIMENT VISUALIZATIONS
# =============================================================================

def plot_sentiment_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create bar chart of sentiment distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    
    sentiment = df['AISent'].value_counts()
    order = ['Very favorable', 'Favorable', 'Indifferent', 'Unsure', 
             'Unfavorable', 'Very unfavorable']
    sentiment = sentiment.reindex([o for o in order if o in sentiment.index])
    
    colors = [COLORS['success'], '#7CB342', COLORS['warning'], 
              COLORS['info'], '#FF7043', COLORS['danger']]
    
    bars = ax.bar(range(len(sentiment)), sentiment.values, color=colors[:len(sentiment)],
                  edgecolor='white', linewidth=1)
    
    # Add value labels
    for bar, val in zip(bars, sentiment.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f'{val:,}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xticks(range(len(sentiment)))
    ax.set_xticklabels(sentiment.index, rotation=30, ha='right')
    ax.set_ylabel('Number of Respondents')
    ax.set_title('AI Tool Sentiment Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_sentiment_by_status(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create stacked bar chart of sentiment by AI status.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    # Cross-tabulation
    crosstab = pd.crosstab(df['AI_Status'], df['Sentiment_Category'], normalize='index') * 100
    
    # Reorder
    status_order = ['Using AI', 'Planning to Use', 'Not Using']
    crosstab = crosstab.reindex([s for s in status_order if s in crosstab.index])
    
    # Plot
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    crosstab[['Positive', 'Neutral', 'Negative']].plot(
        kind='barh', stacked=True, ax=ax, color=colors, edgecolor='white'
    )
    
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Sentiment Distribution by AI Adoption Status', fontsize=14, fontweight='bold')
    ax.legend(title='Sentiment', loc='lower right')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# BENEFITS & CHALLENGES VISUALIZATIONS
# =============================================================================

def plot_benefits(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create horizontal bar chart of AI benefits.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    ai_users = df[df['Is_AI_User'] == 1]
    
    benefits = {
        'Increase Productivity': ai_users['Ben_Increase_productivit'].mean() * 100,
        'Speed Up Learning': ai_users['Ben_Speed_up_learning'].mean() * 100,
        'Greater Efficiency': ai_users['Ben_Greater_efficiency'].mean() * 100,
        'Improve Accuracy': ai_users['Ben_Improve_accuracy_in_'].mean() * 100,
        'Manageable Workload': ai_users['Ben_Make_workload_more_m'].mean() * 100,
        'Improve Collaboration': ai_users['Ben_Improve_collaboratio'].mean() * 100
    }
    
    benefits_df = pd.DataFrame.from_dict(benefits, orient='index', columns=['Percentage'])
    benefits_df = benefits_df.sort_values('Percentage', ascending=True)
    
    bars = ax.barh(range(len(benefits_df)), benefits_df['Percentage'].values, 
                   color=COLORS['success'], edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, benefits_df['Percentage'].values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(benefits_df)))
    ax.set_yticklabels(benefits_df.index)
    ax.set_xlabel('% of AI Users')
    ax.set_title('Top Benefits of AI Coding Tools', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 100)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_challenges(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create horizontal bar chart of AI challenges.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    ai_users = df[df['Is_AI_User'] == 1]
    
    challenges = {
        'Trust Issues': ai_users['Challenge_Trust'].mean() * 100,
        'Lacks Codebase Context': ai_users['Challenge_Context'].mean() * 100,
        'Security Concerns': ai_users['Challenge_Security'].mean() * 100,
        'Training Gaps': ai_users['Challenge_Training'].mean() * 100,
        'Uneven Adoption': ai_users['Challenge_Adoption'].mean() * 100
    }
    
    challenges_df = pd.DataFrame.from_dict(challenges, orient='index', columns=['Percentage'])
    challenges_df = challenges_df.sort_values('Percentage', ascending=True)
    
    bars = ax.barh(range(len(challenges_df)), challenges_df['Percentage'].values, 
                   color=COLORS['danger'], edgecolor='white')
    
    # Add value labels
    for bar, val in zip(bars, challenges_df['Percentage'].values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(challenges_df)))
    ax.set_yticklabels(challenges_df.index)
    ax.set_xlabel('% of AI Users')
    ax.set_title('Top Challenges with AI Coding Tools', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 70)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# ROI VISUALIZATIONS
# =============================================================================

def plot_roi_by_company_size(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create box plot of ROI by company size.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_LARGE)
    
    # Filter valid ROI values
    df_roi = df[df['ROI_Mid'].notna() & (df['ROI_Mid'] < 50000)]  # Cap at 50000% for visibility
    
    size_order = ['Micro (2-9)', 'Small (10-19)', 'Small-Mid (20-99)', 
                  'Mid (100-499)', 'Mid-Large (500-999)', 'Large (1K-5K)', 
                  'Large (5K-10K)', 'Enterprise (10K+)']
    
    df_roi = df_roi[df_roi['Company_Size_Category'].isin(size_order)]
    
    # Create box plot
    sns.boxplot(data=df_roi, x='Company_Size_Category', y='ROI_Mid', 
                order=size_order, palette=PALETTE, ax=ax)
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel('ROI (%)')
    ax.set_xlabel('Company Size')
    ax.set_title('ROI Distribution by Company Size\n(Moderate Scenario: 5 hrs/week saved)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


def plot_roi_scenarios(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create comparison chart of ROI scenarios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    
    scenarios = ['Conservative\n(2 hrs/wk)', 'Moderate\n(5 hrs/wk)', 'Optimistic\n(8 hrs/wk)']
    medians = [df['ROI_Low'].median(), df['ROI_Mid'].median(), df['ROI_High'].median()]
    
    colors = [COLORS['warning'], COLORS['success'], COLORS['primary']]
    bars = ax.bar(scenarios, medians, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, val in zip(bars, medians):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                f'{val:,.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Median ROI (%)')
    ax.set_title('ROI by Productivity Scenario', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# THREAT PERCEPTION VISUALIZATIONS
# =============================================================================

def plot_threat_perception(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create pie chart of job threat perception.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_SMALL)
    
    threat = df['Threat_Category'].value_counts()
    colors = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    
    wedges, texts, autotexts = ax.pie(
        threat.values, 
        labels=threat.index,
        autopct='%1.1f%%',
        colors=colors,
        explode=(0.02, 0.02, 0.02),
        startangle=90
    )
    
    plt.setp(autotexts, size=11, weight='bold')
    ax.set_title('Do Developers See AI as a Job Threat?', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# REGIONAL VISUALIZATIONS
# =============================================================================

def plot_adoption_by_region(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create horizontal bar chart of adoption by region.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE_MEDIUM)
    
    adoption = df.groupby('Region')['Is_AI_User'].agg(['mean', 'count'])
    adoption['Adoption_Rate'] = adoption['mean'] * 100
    adoption = adoption[adoption['count'] >= 500]  # Min sample size
    adoption = adoption.sort_values('Adoption_Rate', ascending=True)
    
    bars = ax.barh(range(len(adoption)), adoption['Adoption_Rate'].values, 
                   color=COLORS['primary'], edgecolor='white')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, adoption['Adoption_Rate'].values)):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center', fontsize=9)
    
    ax.set_yticks(range(len(adoption)))
    ax.set_yticklabels(adoption.index)
    ax.set_xlabel('Adoption Rate (%)')
    ax.set_title('AI Tool Adoption by Region', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 75)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig, ax


# =============================================================================
# DASHBOARD CREATION
# =============================================================================

def create_dashboard(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create a comprehensive dashboard with multiple charts.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    save_path : str, optional
        Path to save the figure
    """
    fig = plt.figure(figsize=(20, 16))
    
    # Title
    fig.suptitle('GenAI ROI Analysis Dashboard\nStack Overflow Developer Survey 2024', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # 1. Adoption Overview (pie)
    ax1 = fig.add_subplot(gs[0, 0])
    adoption = df['AI_Status'].value_counts()
    colors = [COLORS['success'], COLORS['danger'], COLORS['warning']]
    ax1.pie(adoption.values, labels=adoption.index, autopct='%1.1f%%', colors=colors)
    ax1.set_title('AI Adoption Status', fontweight='bold')
    
    # 2. Adoption by Company Size
    ax2 = fig.add_subplot(gs[0, 1])
    adoption_size = df.groupby('Company_Size_Category')['Is_AI_User'].mean() * 100
    size_order = ['Micro (2-9)', 'Small (10-19)', 'Small-Mid (20-99)', 'Mid (100-499)', 
                  'Mid-Large (500-999)', 'Large (1K-5K)', 'Enterprise (10K+)']
    adoption_size = adoption_size.reindex([s for s in size_order if s in adoption_size.index])
    ax2.bar(range(len(adoption_size)), adoption_size.values, color=COLORS['primary'])
    ax2.set_xticks(range(len(adoption_size)))
    ax2.set_xticklabels(['Micro', 'Small', 'Sm-Mid', 'Mid', 'Mid-Lg', 'Large', 'Entrp'], 
                        rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Adoption %')
    ax2.set_title('Adoption by Company Size', fontweight='bold')
    
    # 3. Sentiment Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sentiment = df['Sentiment_Category'].value_counts()
    colors_sent = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    ax3.pie(sentiment.values, labels=sentiment.index, autopct='%1.1f%%', colors=colors_sent)
    ax3.set_title('Sentiment Distribution', fontweight='bold')
    
    # 4. Benefits
    ax4 = fig.add_subplot(gs[1, 0])
    ai_users = df[df['Is_AI_User'] == 1]
    benefits = {
        'Productivity': ai_users['Ben_Increase_productivit'].mean() * 100,
        'Learning': ai_users['Ben_Speed_up_learning'].mean() * 100,
        'Efficiency': ai_users['Ben_Greater_efficiency'].mean() * 100,
        'Accuracy': ai_users['Ben_Improve_accuracy_in_'].mean() * 100
    }
    ax4.barh(list(benefits.keys()), list(benefits.values()), color=COLORS['success'])
    ax4.set_xlabel('% of AI Users')
    ax4.set_title('Top Benefits', fontweight='bold')
    ax4.set_xlim(0, 100)
    
    # 5. Challenges
    ax5 = fig.add_subplot(gs[1, 1])
    challenges = {
        'Trust Issues': ai_users['Challenge_Trust'].mean() * 100,
        'Lacks Context': ai_users['Challenge_Context'].mean() * 100,
        'Security': ai_users['Challenge_Security'].mean() * 100,
        'Training': ai_users['Challenge_Training'].mean() * 100
    }
    ax5.barh(list(challenges.keys()), list(challenges.values()), color=COLORS['danger'])
    ax5.set_xlabel('% of AI Users')
    ax5.set_title('Top Challenges', fontweight='bold')
    ax5.set_xlim(0, 70)
    
    # 6. ROI Scenarios
    ax6 = fig.add_subplot(gs[1, 2])
    scenarios = ['Low\n(2hr/wk)', 'Mid\n(5hr/wk)', 'High\n(8hr/wk)']
    medians = [df['ROI_Low'].median(), df['ROI_Mid'].median(), df['ROI_High'].median()]
    ax6.bar(scenarios, medians, color=[COLORS['warning'], COLORS['success'], COLORS['primary']])
    ax6.set_ylabel('Median ROI %')
    ax6.set_title('ROI by Scenario', fontweight='bold')
    
    # 7. Adoption by Role
    ax7 = fig.add_subplot(gs[2, 0:2])
    adoption_role = df.groupby('Role_Category')['Is_AI_User'].mean() * 100
    adoption_role = adoption_role.drop('Unknown', errors='ignore').sort_values(ascending=True)
    ax7.barh(range(len(adoption_role)), adoption_role.values, color=COLORS['primary'])
    ax7.set_yticks(range(len(adoption_role)))
    ax7.set_yticklabels(adoption_role.index, fontsize=9)
    ax7.set_xlabel('Adoption Rate %')
    ax7.set_title('Adoption by Job Role', fontweight='bold')
    
    # 8. Threat Perception
    ax8 = fig.add_subplot(gs[2, 2])
    threat = df['Threat_Category'].value_counts()
    colors_threat = [COLORS['success'], COLORS['warning'], COLORS['danger']]
    ax8.pie(threat.values, labels=threat.index, autopct='%1.1f%%', colors=colors_threat)
    ax8.set_title('Job Threat Perception', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# GENERATE ALL VISUALIZATIONS
# =============================================================================

def generate_all_visualizations(df: pd.DataFrame, output_dir: str = 'visualizations/'):
    """
    Generate and save all visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Cleaned survey dataframe
    output_dir : str
        Directory to save visualizations
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating all visualizations...")
    
    # Generate each chart
    plot_adoption_overview(df, f'{output_dir}01_adoption_overview.png')
    plot_adoption_by_company_size(df, f'{output_dir}02_adoption_by_company_size.png')
    plot_adoption_by_role(df, f'{output_dir}03_adoption_by_role.png')
    plot_sentiment_distribution(df, f'{output_dir}04_sentiment_distribution.png')
    plot_sentiment_by_status(df, f'{output_dir}05_sentiment_by_status.png')
    plot_benefits(df, f'{output_dir}06_benefits.png')
    plot_challenges(df, f'{output_dir}07_challenges.png')
    plot_roi_scenarios(df, f'{output_dir}08_roi_scenarios.png')
    plot_roi_by_company_size(df, f'{output_dir}09_roi_by_company_size.png')
    plot_threat_perception(df, f'{output_dir}10_threat_perception.png')
    plot_adoption_by_region(df, f'{output_dir}11_adoption_by_region.png')
    create_dashboard(df, f'{output_dir}12_dashboard.png')
    
    print(f"\n✅ All visualizations saved to {output_dir}")
    
    plt.close('all')


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Load cleaned data
    df = pd.read_csv("data/processed/genai_roi_clean.csv")
    
    # Generate all visualizations
    generate_all_visualizations(df, 'visualizations/')
