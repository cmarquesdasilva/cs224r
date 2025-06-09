import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os

# Add the parent directory to the path so we can import constants.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FACET_TO_ITEMS, DOMAIN_TO_FACETS

# Enable progress bar for pandas operations
tqdm.pandas()

def compute_facet_score(row, column_index_to_key):
    """
    Compute the score for each facet based on the responses.
    The function takes a row of responses and computes the score for each facet.
    
    Args:
        row: A row of responses from the dataset
        column_index_to_key: Dictionary mapping column indices to scoring keys
    
    Returns:
        dict: A dictionary containing scores for each facet
    """
    scores = {}
    for facet, items in FACET_TO_ITEMS.items():
        facet_score = 0
        for item in items:
            response = row[item]
            key = column_index_to_key.get(item, 1)  # Default key to 1 if not found
            if key == -1:
                # Reverse scoring for negative keys
                facet_score += (6 - response)
            else:
                facet_score += response

        scores[facet] = facet_score / len(items)
    return scores

def compute_domain_score(row):
    """
    Compute the score for each domain based on the facet scores.
    
    Args:
        row: A row containing facet scores
    
    Returns:
        dict: A dictionary containing scores for each domain
    """
    domain_scores = {}
    for domain, facets in DOMAIN_TO_FACETS.items():
        domain_score = 0
        for facet in facets:
            domain_score += row[facet]
        domain_scores[domain] = domain_score / len(facets)
    return domain_scores

def categorize_age_sex(row):
    """
    Categorize a person by age and sex group.
    
    Args:
        row: A row containing age and sex information
    
    Returns:
        str: The age-sex group category
    """
    # Age group
    age = row['age']
    if age < 18:
        age_group = "Teen"
    elif age <= 24:
        age_group = "Young Adult"
    elif age <= 49:
        age_group = "Adult"
    elif age <= 64:
        age_group = "Older Adult"
    else:
        age_group = "Senior"

    # Sex (assuming 1 = Male, 2 = Female)
    sex = row['sex']
    sex_label = "Male" if sex == 1 else "Female"

    return f"{sex_label}, {age_group}"

def categorize_percentile(value, percentiles):
    """
    Categorize a value based on its position relative to the provided percentiles.
    
    Args:
        value (float): The value to categorize.
        percentiles (list): A list of percentiles [10th, 30th, 70th, 90th].
    
    Returns:
        str: The category ("Very Low", "Low", "Medium", "High", "Very High").
    """
    if value < percentiles[0]:
        return "Very Low"
    elif value < percentiles[1]:
        return "Low"
    elif value < percentiles[2]:
        return "Medium"
    elif value < percentiles[3]:
        return "High"
    else:
        return "Very High"

def load_raw_data(data_path, map_path):
    """
    Load raw data from CSV files.
    
    Args:
        data_path: Path to the IPIP_NEO_120 data file
        map_path: Path to the mpi_120 mapping file
    
    Returns:
        tuple: (data DataFrame, column_index_to_key dictionary)
    """
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    print(f"Loading mapping data from {map_path}")
    map_120 = pd.read_csv(map_path)
    map_120['index'] = map_120.index
    map_120['column_index'] = map_120['index'].apply(lambda x: f"i{x+1}")
    column_index_to_key = map_120.set_index('column_index')['key'].to_dict()
    
    return data, column_index_to_key

def compute_scores(data, column_index_to_key):
    """
    Compute facet and domain scores for the dataset.
    
    Args:
        data: DataFrame containing raw responses
        column_index_to_key: Dictionary mapping column indices to scoring keys
    
    Returns:
        DataFrame: Data with computed scores added
    """
    print("Computing facet scores...")

    # Apply the compute_facet_score function with the column_index_to_key parameter
    data = data.join(data.progress_apply(
        lambda row: compute_facet_score(row, column_index_to_key), 
        axis=1).apply(pd.Series))
    
    print("Computing domain scores...")
    # Apply the compute_domain_score function
    data = data.join(data.progress_apply(compute_domain_score, axis=1).apply(pd.Series))
    
    return data

def add_demographic_groups(data):
    """
    Add age-sex group classifications to the data.
    
    Args:
        data: DataFrame with age and sex information
    
    Returns:
        DataFrame: Data with age_sex_group column added
    """
    print("Adding demographic group classifications...")
    data['age_sex_group'] = data.progress_apply(categorize_age_sex, axis=1)
    return data

def compute_percentiles(data):
    """
    Compute percentiles for each demographic group.
    
    Args:
        data: DataFrame with scores and demographic groups
    
    Returns:
        DataFrame: Percentile data for each demographic group
    """
    print("Computing percentiles by demographic group...")
    percentile_data = []

    for group, group_data in data.groupby('age_sex_group'):
        group_percentiles = {'age_sex_group': group}
        
        # Compute percentiles for facets
        for facet in FACET_TO_ITEMS.keys():
            percentiles = np.percentile(group_data[facet], [10, 30, 70, 90])
            group_percentiles.update({
                f'{facet}_10th': percentiles[0],
                f'{facet}_30th': percentiles[1],
                f'{facet}_70th': percentiles[2],
                f'{facet}_90th': percentiles[3],
            })
        
        # Compute percentiles for domains
        for domain in DOMAIN_TO_FACETS.keys():
            percentiles = np.percentile(group_data[domain], [10, 30, 70, 90])
            group_percentiles.update({
                f'{domain}_10th': percentiles[0],
                f'{domain}_30th': percentiles[1],
                f'{domain}_70th': percentiles[2],
                f'{domain}_90th': percentiles[3],
            })
        
        percentile_data.append(group_percentiles)

    return pd.DataFrame(percentile_data)

def categorize_scores(data, percentile_data):
    """
    Categorize scores based on demographic-specific percentiles.
    
    Args:
        data: DataFrame with computed scores
        percentile_data: DataFrame with percentile thresholds by demographic group
    
    Returns:
        DataFrame: Data with categorized scores
    """
    print("Categorizing scores based on percentiles...")
    for group, group_data in percentile_data.groupby('age_sex_group'):
        # Filter the original dataset for the current group
        group_indices = data['age_sex_group'] == group
        
        # Skip empty groups
        if not any(group_indices):
            continue

        # Categorize facets
        for facet in FACET_TO_ITEMS.keys():
            percentiles = [
                group_data[f'{facet}_10th'].values[0],
                group_data[f'{facet}_30th'].values[0],
                group_data[f'{facet}_70th'].values[0],
                group_data[f'{facet}_90th'].values[0],
            ]
            data.loc[group_indices, f'{facet}_level'] = data.loc[group_indices, facet].apply(
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
            data.loc[group_indices, f'{domain}_level'] = data.loc[group_indices, domain].apply(
                lambda x: categorize_percentile(x, percentiles)
            )
    
    return data