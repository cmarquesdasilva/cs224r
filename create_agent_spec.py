"""
Persona Generator

This script generates persona specifications based on personality trait data.
It processes personality trait data, builds appropriate prompts, and uses language
models to generate detailed persona specifications matching those traits.

The script supports generating personas with different model providers (OpenAI or Together AI),
saving the results, and tracking progress with logging. Model settings are read from
environment variables.

Usage:
    # Set model settings in .env file:
    # MODEL_TYPE=openai
    # MODEL_NAME=gpt-4o
    
    # Run with all data:
    python create_agent_spec.py --output-dir dataset/personas
    
    # Run with sample size:
    python create_agent_spec.py --output-dir dataset/personas --sample-size 20
    
    # Include only specific case IDs:
    python create_agent_spec.py --output-dir dataset/personas --include-file path/to/cases.json
"""

import os
import json
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Any, Set, Optional, Union
from dotenv import load_dotenv

# Import from our refactored modules
from src.model_inference_assets.prompt_builder import PromptBuilder
from src.model_inference_assets.inference_utils import invoke_chat_model
from src.model_inference_assets.helpers import (
    perturb_temperature,
    extract_json_from_response,
    get_processed_case_ids,
    get_included_case_ids
)

# Load environment variables
load_dotenv()

# Configuration constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_SAMPLE_SIZE = 20

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_logging(log_file_path: str) -> None:
    """
    Set up file logging in addition to console logging.
    
    Args:
        log_file_path: Path where to save the log file
    """
    # Create file handler
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_file_path}")

def filter_dataset(
    data: pd.DataFrame, 
    processed_case_ids: Set[int], 
    include_case_ids: Optional[Set[int]] = None
) -> pd.DataFrame:
    """
    Filter the dataset to exclude already processed cases and optionally include specific cases.
    
    Args:
        data: Original dataset
        processed_case_ids: Set of case IDs that have already been processed
        include_case_ids: Optional set of case IDs to include (if provided, only these will be included)
        
    Returns:
        Filtered dataset
    """
    # Remove already processed cases
    filtered_data = data[~data["case"].isin(processed_case_ids)]
    
    # If we have specific cases to include, filter for only those
    if include_case_ids is not None and len(include_case_ids) > 0:
        filtered_data = filtered_data[filtered_data["case"].isin(include_case_ids)]
        
    return filtered_data

def generate_personas(
    data: pd.DataFrame,
    template_path: str,
    output_dir: str,
    all_personas: Optional[List[Dict[str, Any]]] = None,
    model_type: str = "openai",
    model_name: Optional[str] = None,
    system_prompt: str = "",
    temperature: float = DEFAULT_TEMPERATURE
) -> List[Dict[str, Any]]:
    """
    Generate personas using the specified model and data.
    
    Args:
        data: DataFrame containing personality trait data
        template_path: Path to the mustache template for prompt generation
        output_dir: Directory to save generated personas and logs
        model_type: Type of model to use ("openai" or "together")
        model_name: Specific model name to use (optional)
        system_prompt: System prompt to use for OpenAI models
        temperature: Temperature parameter for model inference
        
    Returns:
        List of generated persona specifications
    """      # Set up output paths
    final_output_path = os.path.join(output_dir, "agent_spec.json")
    debug_dir = os.path.join(output_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Initialize PromptBuilder
    prompt_builder = PromptBuilder(data, template_path)
    
    # Generate personas
    if all_personas is None:
        all_personas = []
    processed_ids = set(p.get("case_id") for p in all_personas if "case_id" in p)

    # Set description for progress bar
    desc = f"Generating personas with {model_type.capitalize()}"

    for index in tqdm(range(len(data)), desc=desc):
        case_id = int(data.iloc[index]["case"])

        # Skip already processed cases
        if case_id in processed_ids:
            logger.info(f"Skipping already processed case {case_id}")
            continue

        try:
            # Build prompt
            prompt = prompt_builder.build_prompt(index)

            # Add some randomness to temperature for diversity
            current_temp = perturb_temperature(temperature)


            # Generate persona with model
            model_response = invoke_chat_model(
                system_prompt=system_prompt,
                user_prompt=prompt,
                provider=model_type,
                model_name=model_name,
                extract_json=True,
                temperature=current_temp
            )

            # Handle AIMessage or string response
            if hasattr(model_response, 'content'):
                response = model_response.content
            else:
                response = str(model_response)
            persona_data = extract_json_from_response(response)

            # Add case_id to the persona data
            persona_data["case_id"] = case_id

            # Add to collection
            all_personas.append(persona_data)
            processed_ids.add(case_id)
            logger.info(f"Successfully generated persona for case {case_id}")

        except Exception as e:
            logger.error(f"Error processing case {case_id}: {str(e)}")

        # Save intermediate results periodically
        if (index + 1) % 10 == 0 or index == len(data) - 1:
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(all_personas, f, indent=2)
            logger.info(f"Progress saved: {len(all_personas)} personas written to {final_output_path}")

    return all_personas

def get_output_paths(model_type, model_name, output_base_dir):
    """
    Generate standardized output file paths based on model information.
    
    Args:
        model_type: The model provider (openai, together, etc.)
        model_name: The specific model used
        output_base_dir: Base directory for outputs
        
    Returns:
        dict: Dictionary containing standardized paths for various outputs
    """
    # Create model directory path
    model_dir = os.path.join(output_base_dir, model_type, model_name)
    
    # Ensure the directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    return {
        'model_dir': model_dir,
        'personas_file': os.path.join(model_dir, "agent_spec.json"),
        'log_file': os.path.join(model_dir, "generation.log"),
        'debug_dir': os.path.join(model_dir, "debug")
    }

def get_case_ids_to_process(data_df, output_paths, include_file=None):
    """
    Determine which case IDs should be processed in this run, using helpers for processed/included IDs.
    Returns filtered DataFrame and the sets used.
    """
    personas_file = output_paths['personas_file']
    processed_case_ids, all_personas = get_processed_case_ids(personas_file)
    included_case_ids = get_included_case_ids(include_file) if include_file else set()

    logger.info(f"Found {len(processed_case_ids)} already processed cases")
    if included_case_ids:
        logger.info(f"Including {len(included_case_ids)} specific cases from {include_file}")

    # Remove already processed cases
    filtered_data = data_df[~data_df["case"].astype(int).isin(processed_case_ids)]

    # If include_case_ids is provided, ensure those are present in the filtered data
    if included_case_ids:
        included_data = data_df[data_df["case"].astype(int).isin(included_case_ids)]
        filtered_data = pd.concat([filtered_data, included_data]).drop_duplicates(subset="case")

    return filtered_data, included_case_ids, all_personas

def main():
    """
    Main function to parse arguments and run the persona generation pipeline.
    """
    # Load environment variables
    load_dotenv()
    
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Generate persona specifications from personality traits')
    parser.add_argument('--system-prompt', type=str, default="",
                      help='Optional system prompt to use')
    parser.add_argument('--template-path', type=str, default="src/templates/persona_template.mustache",
                      help='Path to the mustache template for prompt generation')
    parser.add_argument('--data-path', type=str, default="dataset/test_set.csv",
                      help='Path to the dataset containing personality trait data')
    parser.add_argument('--output-dir', type=str, default='dataset/agent_spec/',
                      help='Base directory to save the generated personas')
    parser.add_argument('--sample-size', type=int, default=1,
                      help='Number of samples to take (0 = use all available data)')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                      help='Temperature parameter for model inference')
    parser.add_argument('--include-file', type=str, default="",
                      help='Path to JSON file with case IDs to include (if provided, only these cases will be processed)')
    
    args = parser.parse_args()
    
    # Get model configuration from environment variables
    model_type = os.environ.get("MODEL_TYPE", "openai")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o")
    
    # Generate standardized output paths
    output_paths = get_output_paths(model_type, model_name, args.output_dir)
    
    # Set up logging with file
    setup_logging(output_paths['log_file'])
    
    # Log model configuration
    logger.info(f"Using model type: {model_type}")
    logger.info(f"Using model name: {model_name}")
    
    # Load data
    logger.info(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    
    # Get cases to process and sets for logic
    filtered_data, included_case_ids, all_personas = get_case_ids_to_process(
        data, output_paths, args.include_file
    )

    if args.sample_size > 0 and len(filtered_data) > args.sample_size:
        if included_case_ids:
            included_mask = filtered_data["case"].astype(int).isin(included_case_ids)
        else:
            included_mask = pd.Series([False] * len(filtered_data), index=filtered_data.index)

        included_rows = filtered_data[included_mask]
        remaining_rows = filtered_data[~included_mask]
        sample_size = max(0, args.sample_size - len(included_rows))

        if sample_size > 0 and len(remaining_rows) > 0:
            sampled = remaining_rows.sample(sample_size, random_state=42)
            filtered_data = pd.concat([included_rows, sampled])
        else:
            filtered_data = included_rows

    logger.info(f"Sampled {len(filtered_data)} cases (including {len(included_rows)} required cases)")
    
    # Run persona generation
    logger.info(f"Starting persona generation with {model_type}/{model_name}")
    all_personas = list(all_personas) if all_personas is not None else []
    all_personas = generate_personas(
        data=filtered_data,
        template_path=args.template_path,
        output_dir=output_paths['model_dir'],
        all_personas=all_personas,
        model_type=model_type,
        model_name=model_name,
        system_prompt=args.system_prompt,
        temperature=args.temperature
    )

    # Save all personas (append mode already handled)
    with open(output_paths['personas_file'], 'w', encoding='utf-8') as f:
        json.dump(all_personas, f, indent=2)
    logger.info(f"Persona generation complete. Total: {len(all_personas)} personas.")

if __name__ == "__main__":
    main()