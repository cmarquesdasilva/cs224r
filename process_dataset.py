import os
import sys
from src.data_processing.transform_dataset import (
    load_raw_data,
    compute_scores,
    add_demographic_groups, 
    compute_percentiles,
    categorize_scores
)

def main():
    """
    Run the complete data processing workflow.
    """
    # Define file paths
    data_dir = os.path.join(os.path.dirname(__file__), "dataset")
    raw_dir = os.path.join(data_dir, "raw")
    processed_dir = os.path.join(data_dir, "processed")
    
    # Create directories if they don't exist
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Define input and output file paths
    raw_data_path = os.path.join(raw_dir, "IPIP_NEO_120.csv")
    map_data_path = os.path.join(raw_dir, "mpi_120.csv")
    scores_output_path = os.path.join(processed_dir, "IPIP_NEO_120_with_scores.csv")
    percentiles_output_path = os.path.join(processed_dir, "percentile_data.csv")
    final_output_path = os.path.join(processed_dir, "IPIP_NEO_120_processed.csv")
    
    # Step 1: Load the raw data
    data, column_index_to_key = load_raw_data(raw_data_path, map_data_path)
    
    # Step 2: Compute facet and domain scores
    data = compute_scores(data, column_index_to_key)
    
    # Step 3: Add demographic group classifications
    data = add_demographic_groups(data)
    
    # Step 4: Save intermediate results
    data.to_csv(scores_output_path, index=False)
    
    # Step 5: Compute percentiles for each demographic group
    percentile_data = compute_percentiles(data)
    
    # Step 6: Save percentile data
    percentile_data.to_csv(percentiles_output_path, index=False)
    
    # Step 7: Categorize scores based on percentiles
    data = categorize_scores(data, percentile_data)
    
    # Step 8: Save final processed data
    data.to_csv(final_output_path, index=False)
        
if __name__ == "__main__":
    main()