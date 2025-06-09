"""
Personality Assessment Evaluation Module.

This module evaluates agreement between baseline and processed personality assessment data
from the IPIP-NEO-120 personality inventory. It calculates accuracy metrics at various levels:
individual questions, facets, domains, and agents.

The script performs the following operations:
- Loading and merging data from multiple sources
- Computing personality facet and domain scores
- Categorizing scores based on age and sex group percentiles
- Evaluating agreement between baseline and processed assessments
- Calculating accuracy metrics at different granularity levels
"""

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.evaluation.evaluation_metrics import (
    compute_scores,
    categorize_levels,
    evaluate_agreement,
    compute_score_alignment,
    calculate_agent_accuracy,
    plot_agreement_heatmaps,
)

# Enable progress bars for pandas operations
tqdm.pandas()

# Constants for file paths
BASELINE_PATHS = [
    r'results/trained_qwen_mpi_120_results.csv', 
    r'results/baseline_qwen_7b_mpi_120_results.csv',
    r'results/baseline_gpt4o_mpi_120_results.csv'
]

PROCESSED_PATH = r'dataset/processed/IPIP_NEO_120_processed.csv'
PERCENTILE_PATH = r'dataset/processed/percentile_data.csv'

def get_model_name(file_path: str) -> str:
    """
    Extract model name from file path by taking the part before "_mpi_120_results.csv".
    
    Args:
        file_path: Path to the baseline results file
        
    Returns:
        str: Model name extracted from the file path
    """
    # Get just the filename without directory
    file_name = os.path.basename(file_path)
    # Extract the part before "_mpi_120_results.csv"
    model_name = file_name.split("_mpi_120_results.csv")[0]
    return model_name

def load_data(baseline_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load baseline, processed, and percentile data from CSV files.
    
    Args:
        baseline_path: Path to the specific baseline data file to load
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - baseline_data: Experimental results data
            - processed_data: Processed IPIP NEO assessment data
            - percentile_data: Percentile reference data for normalization
    """
    baseline_data = pd.read_csv(baseline_path)
    processed_data = pd.read_csv(PROCESSED_PATH)
    percentile_data = pd.read_csv(PERCENTILE_PATH)
    return baseline_data, processed_data, percentile_data

def merge_data(baseline_data: pd.DataFrame, processed_data: pd.DataFrame) -> pd.DataFrame:
    """
    Merge baseline data with processed data based on case ID.
    
    Args:
        baseline_data: Dataframe containing baseline assessment results
        processed_data: Dataframe containing processed assessment results
    
    Returns:
        pd.DataFrame: Merged dataframe with demographic information added to baseline data
    """
    return baseline_data.merge(
        processed_data[['case', 'age_sex_group']],
        on='case',
        how='left'
    )

def process_single_baseline(baseline_path: str) -> Tuple[Dict[str, float], float, Dict[str, float], float, Dict[str, float]]:
    """
    Process a single baseline file and compute score alignment and accuracy.
    
    Args:
        baseline_path: Path to the baseline data file
        
    Returns:
        Tuple[Dict[str, float], float, Dict[str, float], float, Dict[str, float]]: 
            - Agent alignment scores dictionary
            - Overall average alignment score
            - Agent accuracy scores dictionary
            - Overall average accuracy
            - Trait and facet agreement dictionary
    """
    print(f"\nProcessing {baseline_path}...")
    model_name = get_model_name(baseline_path)
    print(f"Model: {model_name}")
    
    # Load data
    baseline_data, processed_data, percentile_data = load_data(baseline_path)
    
    # Prepare and process data
    baseline_data = merge_data(baseline_data, processed_data)
    baseline_data = compute_scores(baseline_data)
    baseline_data = categorize_levels(baseline_data, percentile_data)
    
    # Evaluate and get comparison data
    comparison_data, trait_facet_agreement = evaluate_agreement(baseline_data, processed_data)
    
    # Compute score alignment
    alignment_scores, overall_avg = compute_score_alignment(comparison_data)
    
    # Compute accuracy
    agent_accuracies = calculate_agent_accuracy(comparison_data)
    overall_accuracy = sum(agent_accuracies.values()) / len(agent_accuracies) if agent_accuracies else 0
    
    return alignment_scores, overall_avg, agent_accuracies, overall_accuracy, trait_facet_agreement

def main() -> None:
    """
    Main function to orchestrate the evaluation workflow.
    
    Executes the full pipeline for personality assessment evaluation:
    1. Processing multiple baseline files
    2. Computing score alignment and accuracy for each file
    3. Compiling results into comparative dataframes
    4. Computing and printing average scores for each model
    5. Visualizing trait and facet agreement with heatmaps
    """
    # Dictionaries to store data by model
    all_alignment_scores: Dict[str, Dict[str, float]] = {}
    all_accuracy_scores: Dict[str, Dict[str, float]] = {}
    model_alignment_averages: Dict[str, float] = {}
    model_accuracy_averages: Dict[str, float] = {}
    all_trait_facet_agreements: Dict[str, Dict[str, float]] = {}
    
    # Process each baseline file
    for baseline_path in BASELINE_PATHS:
        model_name = get_model_name(baseline_path)
        alignment_scores, overall_alignment_avg, accuracy_scores, overall_accuracy_avg, trait_facet_agreement = process_single_baseline(baseline_path)
        
        all_alignment_scores[model_name] = alignment_scores
        all_accuracy_scores[model_name] = accuracy_scores
        model_alignment_averages[model_name] = overall_alignment_avg
        model_accuracy_averages[model_name] = overall_accuracy_avg
        all_trait_facet_agreements[model_name] = trait_facet_agreement
    
    # Create a DataFrame to compare alignment scores across models
    # First, get all unique agent IDs
    all_alignment_agents = set()
    for scores in all_alignment_scores.values():
        all_alignment_agents.update(scores.keys())
    
    # Create the alignment DataFrame
    alignment_df = pd.DataFrame(index=sorted(all_alignment_agents))
    
    # Fill in the DataFrame with alignment scores for each model
    for model, scores in all_alignment_scores.items():
        alignment_df[model] = pd.Series(scores)
    
    # Print the comparative alignment DataFrame
    print("\n\nModel Comparison - Score Alignment by Agent (lower is better):")
    print(alignment_df)
    
    # Print average alignment score for each model
    print("\nAverage Score Alignment by Model (lower is better):")
    for model, avg in model_alignment_averages.items():
        print(f"{model}: {avg:.4f}")
    
    # Create a DataFrame to compare accuracy scores across models
    all_accuracy_agents = set()
    for scores in all_accuracy_scores.values():
        all_accuracy_agents.update(scores.keys())
    
    # Create the accuracy DataFrame
    accuracy_df = pd.DataFrame(index=sorted(all_accuracy_agents))
    
    # Fill in the DataFrame with accuracy scores for each model
    for model, scores in all_accuracy_scores.items():
        accuracy_df[model] = pd.Series(scores)
    
    # Print the comparative accuracy DataFrame
    print("\n\nModel Comparison - Accuracy by Agent (higher is better):")
    print(accuracy_df)
    
    # Print average accuracy score for each model
    print("\nAverage Accuracy by Model (higher is better):")
    for model, avg in model_accuracy_averages.items():
        print(f"{model}: {avg:.2%}")
      # Create heatmaps for trait and facet agreement across models
    plot_agreement_heatmaps(all_trait_facet_agreements)
    

if __name__ == "__main__":
    main()
