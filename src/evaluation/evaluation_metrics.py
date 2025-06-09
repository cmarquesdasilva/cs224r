"""
evaluation_metrics.py

Core evaluation metric functions for personality assessment analysis.
These functions are stateless and assume input DataFrames are already
prepared and formatted as expected.
"""

import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.constants import DOMAIN_TO_FACETS, FACET_TO_ITEMS, COLUMN_INDEX_TO_KEY
from src.data_processing.transform_dataset import (
    compute_facet_score,
    categorize_percentile,
    compute_domain_score,
)

def compute_scores(baseline_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute facet and domain scores from raw assessment data.
    
    Applies the scoring functions to each row of the baseline data to calculate
    both facet-level and domain-level personality scores.
    
    Args:
        baseline_data: Dataframe containing raw assessment responses
    
    Returns:
        pd.DataFrame: Dataframe with computed facet and domain scores added as columns
    """
    # Compute facet scores (e.g., anxiety, assertiveness)
    baseline_data = baseline_data.join(
        baseline_data.progress_apply(lambda row: compute_facet_score(row, COLUMN_INDEX_TO_KEY), axis=1).apply(pd.Series)
    )
    
    # Compute domain scores (e.g., neuroticism, extraversion)
    baseline_data = baseline_data.join(
        baseline_data.progress_apply(compute_domain_score, axis=1).apply(pd.Series)
    )
    
    return baseline_data


def categorize_levels(baseline_data: pd.DataFrame, percentile_data: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize facets and domains based on demographic-specific percentiles.
    
    Assigns descriptive levels to scores based on percentile thresholds that vary
    by age and sex group. Levels represent low, average-low, average-high, or high
    scores relative to normative data.
    
    Args:
        baseline_data: Dataframe with computed facet and domain scores
        percentile_data: Reference data with percentile thresholds by demographic group
    
    Returns:
        pd.DataFrame: Dataframe with level categorizations for facets and domains
    """
    for group, group_data in percentile_data.groupby('age_sex_group'):
        # Filter data for current demographic group
        group_indices = baseline_data['age_sex_group'] == group

        # Categorize facets
        for facet in FACET_TO_ITEMS.keys():
            percentiles = [
                group_data[f'{facet}_10th'].values[0],
                group_data[f'{facet}_30th'].values[0],
                group_data[f'{facet}_70th'].values[0],
                group_data[f'{facet}_90th'].values[0],
            ]
            baseline_data.loc[group_indices, f'{facet}_level'] = baseline_data.loc[group_indices, facet].apply(
                lambda x: categorize_percentile(x, percentiles)
            )

        # Categorize domains
        for domain in DOMAIN_TO_FACETS.keys():
            percentiles = [
                group_data[f'{domain}_10th'].values[0],
                group_data[f'{domain}_30th'].values[0],
                group_data[f'{domain}_70th'].values[0],
                group_data[f'{domain}_90th'].values[0],
            ]
            baseline_data.loc[group_indices, f'{domain}_level'] = baseline_data.loc[group_indices, domain].apply(
                lambda x: categorize_percentile(x, percentiles)
            )
            
    return baseline_data


def evaluate_agreement(baseline_data: pd.DataFrame, processed_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Evaluate agreement between baseline and processed data for all traits and facets.
    
    Computes the percentage of matches between baseline and processed categorizations
    for each personality trait and facet.
    
    Args:
        baseline_data: Dataframe with baseline categorized levels
        processed_data: Dataframe with processed categorized levels
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]:
            - Comparison dataframe with both sets of results
            - Dictionary mapping trait/facet names to their agreement percentages
    """
    # Merge datasets with distinct column suffixes
    comparison_data = baseline_data.merge(
        processed_data,
        on='case',
        suffixes=('_baseline', '_processed')
    )

    # Calculate agreement percentages for each trait and facet
    trait_facet_agreement: Dict[str, float] = {}
    for trait_or_facet in list(FACET_TO_ITEMS.keys()) + list(DOMAIN_TO_FACETS.keys()):
        baseline_col = f'{trait_or_facet}_level_baseline'
        processed_col = f'{trait_or_facet}_level_processed'
        agreement = (comparison_data[baseline_col] == comparison_data[processed_col]).mean()
        trait_facet_agreement[trait_or_facet] = agreement

    # Output summary of agreement metrics
    print("Agreement for traits and facets:")
    for key, value in trait_facet_agreement.items():
        print(f"{key}: {value:.2%}")

    return comparison_data, trait_facet_agreement


def calculate_trait_and_facet_accuracy(comparison_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate accuracy metrics for each domain and its constituent facets.
    
    Computes agreement percentages at both domain and facet levels, with domain
    accuracy calculated as the average of its facet accuracies.
    
    Args:
        comparison_data: Dataframe containing both baseline and processed results
    
    Returns:
        Dict[str, float]: Dictionary mapping domains to their accuracy scores
    """
    domain_accuracy: Dict[str, float] = {}
    facet_accuracy: Dict[str, float] = {}

    # Calculate accuracy for each domain's facets
    for domain, facets in DOMAIN_TO_FACETS.items():
        # Calculate accuracy for each facet
        facet_accuracies = []
        
        for facet in facets:
            facet_baseline_col = f'{facet}_level_baseline'
            facet_processed_col = f'{facet}_level_processed'
            accuracy = (comparison_data[facet_baseline_col] == comparison_data[facet_processed_col]).mean()
            facet_accuracy[facet] = accuracy
            facet_accuracies.append(accuracy)

        # Calculate domain accuracy as average of its facets
        domain_accuracy[domain] = np.mean(facet_accuracies)

    # Output hierarchical accuracy results
    print("\nDomain and Facet Accuracy:")
    for domain, accuracy in domain_accuracy.items():
        print(f"Domain {domain}: {accuracy:.2%}")
        for facet in DOMAIN_TO_FACETS[domain]:
            print(f"  Facet {facet}: {facet_accuracy[facet]:.2%}")
            
    return domain_accuracy


def compute_score_alignment(comparison_data: pd.DataFrame) -> Tuple[Dict[str, float], float]:
    """
    Compute the average absolute difference between baseline and processed scores.
    
    For each agent/case, calculates the alignment between baseline and processed
    data by measuring the absolute differences between corresponding columns and
    computing the average difference.
    
    Args:
        comparison_data: Dataframe containing both baseline and processed results
        
    Returns:
        Tuple[Dict[str, float], float]: 
            - Dictionary mapping agent IDs to their average score alignment
              (lower values indicate better alignment)
            - Overall average alignment score
    """
    # Define columns to compare (questions, facets, domains)
    question_cols = [f'i{num}' for num in range(1, 121)]
    facet_cols = list(FACET_TO_ITEMS.keys())
    domain_cols = list(DOMAIN_TO_FACETS.keys())
    
    alignment_scores: Dict[str, float] = {}
    overall_diffs = []
    
    # Group by case/agent and compute alignment
    for case, case_data in comparison_data.groupby('case'):
        total_diffs = []
        
        # Calculate differences for individual questions
        for question in question_cols:
            if f'{question}_baseline' in case_data.columns and f'{question}_processed' in case_data.columns:
                diff = abs(case_data[f'{question}_baseline'] - case_data[f'{question}_processed']).mean()
                total_diffs.append(diff)
        
        # Calculate differences for facets
        for facet in facet_cols:
            if f'{facet}_baseline' in case_data.columns and f'{facet}_processed' in case_data.columns:
                diff = abs(case_data[f'{facet}_baseline'] - case_data[f'{facet}_processed']).mean()
                total_diffs.append(diff)
        
        # Calculate differences for domains
        for domain in domain_cols:
            if f'{domain}_baseline' in case_data.columns and f'{domain}_processed' in case_data.columns:
                diff = abs(case_data[f'{domain}_baseline'] - case_data[f'{domain}_processed']).mean()
                total_diffs.append(diff)
        
        # Compute average alignment score for this agent
        avg_diff = np.mean(total_diffs) if total_diffs else 0
        alignment_scores[case] = avg_diff
        overall_diffs.extend(total_diffs)
    
    # Calculate overall average alignment
    overall_avg = np.mean(overall_diffs) if overall_diffs else 0
    
    # Output alignment results
    print("\nScore Alignment (average absolute difference):")
    print(f"Overall average alignment: {overall_avg:.4f}")
    print("Per-agent alignment (lower is better):")
    for agent, alignment in alignment_scores.items():
        print(f"Agent {agent}: {alignment:.4f}")
    
    return alignment_scores, overall_avg


def calculate_agent_accuracy(comparison_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate accuracy metrics at overall and per-agent levels.
    
    Computes question-level agreement percentages across all agents
    and individually for each agent in the dataset.
    
    Args:
        comparison_data: Dataframe containing both baseline and processed results
    
    Returns:
        Dict[str, float]: Dictionary mapping agent IDs to their accuracy scores
    """
    # Calculate overall accuracy across all questions and agents
    total_questions = 0
    total_agreements = 0

    for question in [f'i{num}' for num in range(1, 121)]:
        total_questions += comparison_data.shape[0]
        total_agreements += (comparison_data[f'{question}_baseline'] == comparison_data[f'{question}_processed']).sum()

    overall_accuracy = total_agreements / total_questions if total_questions > 0 else 0
    print(f"\nOverall accuracy across all questions: {overall_accuracy:.2%}")

    # Calculate per-agent accuracy
    agent_accuracies: Dict[str, float] = {}
    
    for case, case_data in comparison_data.groupby('case'):
        total_questions = 0
        total_agreements = 0

        for question in [f'i{num}' for num in range(1, 121)]:
            total_questions += case_data.shape[0]
            total_agreements += (case_data[f'{question}_baseline'] == case_data[f'{question}_processed']).sum()

        agent_accuracy = total_agreements / total_questions if total_questions > 0 else 0
        agent_accuracies[case] = agent_accuracy

    # Output per-agent accuracy results
    print("\nAccuracy for each agent:")
    for agent, accuracy in agent_accuracies.items():
        print(f"Agent {agent}: {accuracy:.2%}")
        
    return agent_accuracies


def plot_agreement_heatmaps(all_agreements: Dict[str, Dict[str, float]]) -> None:
    """
    Create heatmaps to visualize agreement percentages for facets and domains across models.
    
    Creates two separate heatmaps: one for facets and one for domains (traits),
    showing how well each model performs across all personality metrics.
    
    Args:
        all_agreements: Dictionary mapping model names to their agreement metrics dictionaries
    """
    # Extract agreement data
    models = list(all_agreements.keys())
    facets = list(FACET_TO_ITEMS.keys())
    domains = list(DOMAIN_TO_FACETS.keys())
    
    # Create DataFrames for facets and domains
    facet_data = pd.DataFrame(index=facets, columns=models)
    domain_data = pd.DataFrame(index=domains, columns=models)
    
    # Fill DataFrames with agreement values
    for model, agreements in all_agreements.items():
        for facet in facets:
            if facet in agreements:
                facet_data.loc[facet, model] = agreements[facet]
        
        for domain in domains:
            if domain in agreements:
                domain_data.loc[domain, model] = agreements[domain]
    
    # Convert to numeric and fill NaN values with 0
    facet_data = facet_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    domain_data = domain_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
      # Define a function to set text color based on cell value
    def text_color_function(val):
        return 'black' if 0.3 < val < 0.7 else 'white'    # Plot facet heatmap with improved visibility
    sns.heatmap(
        facet_data, 
        annot=True, 
        cmap="RdBu", 
        fmt=".2%", 
        ax=ax1, 
        cbar_kws={'label': 'Agreement %'},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        linewidths=0.5,
        linecolor='lightgray'
    )
    ax1.set_title("Agreement for Facets Across Models", fontsize=16)
    ax1.set_ylabel("Facets", fontsize=12)
    ax1.set_xlabel("Models", fontsize=12)
    
    # Adjust text colors based on cell color intensity
    for text in ax1.texts:
        value = float(text.get_text().strip('%')) / 100
        text.set_color(text_color_function(value))    # Plot domain heatmap with improved visibility
    sns.heatmap(
        domain_data, 
        annot=True, 
        cmap="RdBu", 
        fmt=".2%", 
        ax=ax2, 
        cbar_kws={'label': 'Agreement %'},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        linewidths=0.5,
        linecolor='lightgray'
    )
    ax2.set_title("Agreement for Personality Domains Across Models", fontsize=16)
    ax2.set_ylabel("Domains", fontsize=12)
    ax2.set_xlabel("Models", fontsize=12)
    
    # Adjust text colors based on cell color intensity
    for text in ax2.texts:
        value = float(text.get_text().strip('%')) / 100
        text.set_color(text_color_function(value))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = r"results/agreement_heatmaps.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nHeatmaps saved to: {output_path}")