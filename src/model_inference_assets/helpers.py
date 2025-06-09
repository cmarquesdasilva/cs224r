"""
Helper utilities for persona generation.

This module contains utility functions used across the persona generation process,
including file operations, JSON handling, and parameter adjustments.
"""
import json
import os
import random
import re
from typing import List, Dict, Any, Union, Optional

def convert_to_proper_json_format(text: str) -> str:
    """
    Clean and format text to ensure it's valid JSON.
    
    Args:
        text: The text containing JSON to clean
        
    Returns:
        Properly formatted JSON string
    """
    # Remove any leading/trailing whitespace
    text = text.strip()
    
    # If text is wrapped in backticks (common in model outputs), remove them
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'^```\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    
    # Remove any additional trailing or leading brackets/braces
    text = text.strip('[]{}')
    
    # Wrap in curly braces if not already present
    if not (text.startswith('{') and text.endswith('}')):
        text = '{' + text + '}'
    
    return text

def extract_json_from_response(response: str) -> Dict[str, Any]:
    """
    Extract JSON object from a model response string.
    
    Args:
        response: String response from a model that contains JSON
        
    Returns:
        Extracted JSON as a dictionary
    
    Raises:
        ValueError: If extraction or parsing fails
    """
    # Method 1: Direct parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Extract from code blocks
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_match = re.search(code_block_pattern, response)
    if code_match:
        try:
            json_str = code_match.group(1).strip()
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Method 3: Extract and clean JSON-like structure
    json_pattern = r"{[\s\S]*?}"
    json_match = re.search(json_pattern, response)
    if json_match:
        try:
            json_str = json_match.group(0)
            
            # Fix common JSON issues
            # Replace single quotes with double quotes
            cleaned = json_str.replace("'", '"')
            # Remove trailing commas
            cleaned = re.sub(r',\s*}', '}', cleaned)
            cleaned = re.sub(r',\s*\]', ']', cleaned)
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    
    # If all methods fail
    raise ValueError(f"Failed to extract valid JSON. Response starts with: {response[:50]}...")

def perturb_temperature(base_temp: float, variance: float = 0.05) -> float:
    """
    Add slight randomness to the temperature parameter for model inference.
    
    Args:
        base_temp: Base temperature value
        variance: Maximum amount to vary temperature (default: 0.05)
    
    Returns:
        Slightly adjusted temperature value
    """
    # Add random adjustment between -variance and +variance
    adjustment = (random.random() * 2 - 1) * variance
    adjusted_temp = base_temp + adjustment
    
    # Ensure temperature stays within reasonable bounds (0.0 to 2.0)
    return max(0.0, min(2.0, adjusted_temp))

def save_persona_to_file(persona_data: Dict[str, Any], output_path: str, case_id: Optional[str] = None) -> str:
    """
    Save a generated persona to a JSON file.
    
    Args:
        persona_data: The persona data to save
        output_path: Directory path where to save the file
        case_id: Optional case ID to use in filename (default: use persona_id from data)
    
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Use provided case_id or extract from persona_data
    file_id = case_id if case_id else persona_data.get('persona_id', f"persona_{random.randint(1000, 9999)}")
    
    # Ensure the file_id is string and doesn't contain problematic characters
    file_id = str(file_id).replace('/', '_').replace('\\', '_')
    
    # Create the output file path
    file_path = os.path.join(output_path, f"{file_id}.json")
    
    # Save the data to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(persona_data, f, indent=2, ensure_ascii=False)
    
    return file_path

def read_persona_from_file(file_path: str) -> Dict[str, Any]:
    """
    Read a persona from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Persona data as a dictionary
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file doesn't contain valid JSON
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def read_json_file(file_path):
    """
    Read a JSON file and return its content, or an empty list if not found.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_processed_case_ids(personas_file):
    """
    Return a set of processed case_ids and the list of all personas from the personas file.
    """
    personas = read_json_file(personas_file)
    return set(int(p.get("case_id")) for p in personas if "case_id" in p), personas

def get_included_case_ids(include_file):
    """
    Return a set of included case_ids from the include file.
    """
    included = read_json_file(include_file)
    if isinstance(included, list) and all(isinstance(x, int) for x in included):
        return set(included)
    elif isinstance(included, list):
        return set(int(p.get("case_id")) for p in included if "case_id" in p)
    return set()