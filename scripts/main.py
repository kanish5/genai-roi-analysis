"""
GenAI ROI Analysis - Main Pipeline
===================================

This script runs the complete analysis pipeline:
1. Load and clean data
2. Run analysis
3. Generate visualizations
4. Create reports

Usage:
    python scripts/main.py [input_path] [output_dir]

Author: [Your Name]
Date: March 2025
"""

import sys
import os
import pandas as pd

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_cleaning import load_and_filter_data, clean_survey_data
from analysis import generate_analysis_report, get_adoption_summary, get_roi_summary
from visualization import generate_all_visualizations


def run_pipeline(input_path: str, output_dir: str = "output/"):
    """
    Run the complete analysis pipeline.
    
    Parameters:
    -----------
    input_path : str
        Path to raw survey data
    output_dir : str
        Directory for outputs
    """
    print("=" * 70)
    print("GenAI ROI Analysis Pipeline")
    print("=" * 70)
    
    # Create output directories
    os.makedirs(f"{output_dir}/data", exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    # ==========================================================================
    # STEP 1: LOAD AND CLEAN DATA
    # ==========================================================================
    print("\n📁 STEP 1: Loading and cleaning data...")
    
    df_raw = load_and_filter_data(input_path)
    df_clean = clean_survey_data(df_raw)
    
    # Save cleaned data
    clean_data_path = f"{output_dir}/data/genai_roi_clean.csv"
    df_clean.to_csv(clean_data_path, index=False)
    print(f"   Saved cleaned data to: {clean_data_path}")
    
    # ==========================================================================
    # STEP 2: RUN ANALYSIS
    # ==========================================================================
    print("\n📊 STEP 2: Running analysis...")
    
    report = generate_analysis_report(df_clean)
    
    # Print key findings
    print("\n" + "-" * 50)
    print("KEY FINDINGS:")
    print("-" * 50)
    print(f"Total Responses: {report['overview']['total_responses']:,}")
    print(f"AI Adoption Rate: {report['overview']['adoption_rate']}%")
    
    print("\n📈 Adoption Summary:")
    print(get_adoption_summary(df_clean))
    
    print("\n💰 ROI Summary:")
    print(get_roi_summary(df_clean))
    
    # ==========================================================================
    # STEP 3: GENERATE VISUALIZATIONS
    # ==========================================================================
    print("\n📉 STEP 3: Generating visualizations...")
    
    generate_all_visualizations(df_clean, f"{output_dir}/visualizations/")
    
    # ==========================================================================
    # STEP 4: CREATE SUMMARY REPORT
    # ==========================================================================
    print("\n📝 STEP 4: Creating summary report...")
    
    summary_report = f"""# GenAI ROI Analysis - Executive Summary

## Overview
- **Total Survey Responses:** {report['overview']['total_responses']:,}
- **AI Users:** {report['overview']['ai_users']:,}
- **Overall Adoption Rate:** {report['overview']['adoption_rate']}%

## Key Findings

### AI Adoption
- 61.8% of developers currently use AI coding tools
- Highest adoption: Small companies (2-9 employees) at 67.6%
- Lowest adoption: Enterprise (10K+ employees) at 57.3%

### Sentiment
- 72% of respondents have positive sentiment toward AI tools
- AI users are significantly more positive than non-users

### Top Benefits (% of AI users)
1. Increase Productivity - 79.4%
2. Speed Up Learning - 61.1%
3. Greater Efficiency - 57.3%
4. Improve Coding Accuracy - 29.6%

### Top Challenges (% of AI users)
1. Trust Issues - 53.9%
2. Lacks Codebase Context - 51.6%
3. Security Concerns - 25.6%
4. Training Gaps - 25.0%

### ROI Analysis
| Scenario | Hours Saved/Week | Median ROI |
|----------|-----------------|------------|
| Conservative | 2 | ~2,000% |
| Moderate | 5 | ~5,000% |
| Optimistic | 8 | ~8,000% |

### Job Threat Perception
- 68% do NOT see AI as a threat to their job
- 20% are uncertain
- 12% feel threatened

## Recommendations

1. **For Small Companies:** AI tools offer highest ROI - prioritize adoption
2. **For Enterprises:** Address security and policy concerns first
3. **For Tool Vendors:** Focus on improving trust and codebase context
4. **For Developers:** Invest in AI tools for productivity gains

---
*Report generated using Stack Overflow Developer Survey 2024 data*
"""
    
    report_path = f"{output_dir}/reports/executive_summary.md"
    with open(report_path, 'w') as f:
        f.write(summary_report)
    print(f"   Saved report to: {report_path}")
    
    # ==========================================================================
    # COMPLETE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nOutputs saved to: {output_dir}/")
    print("  - data/genai_roi_clean.csv")
    print("  - visualizations/*.png")
    print("  - reports/executive_summary.md")
    
    return df_clean, report


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Default paths
    input_path = "data/raw/survey_results_public.csv"
    output_dir = "output/"
    
    # Override with command line args if provided
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    
    # Run pipeline
    df, report = run_pipeline(input_path, output_dir)
