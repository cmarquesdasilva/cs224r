"""
This script prepares a workflow for fine-tuning a model using DPO (Direct Preference Optimization).
It reads two data sources (agent JSON files), iterates through them, and asks a model
which datapoint is more aligned with the prompt specification.
"""

import os
import json
import logging
import pandas as pd
from src.model_inference_assets.prompt_builder import PromptBuilder
from src.model_inference_assets.inference_utils import invoke_chat_model
from src.model_inference_assets.helpers import extract_json_from_response, read_json_file
import random
import chevron  # For rendering mustache templates
from tqdm import tqdm
import re
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("judge_personality.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def map_agents_by_case_id(agents):
    """Return a dict mapping case_id to agent for a list of agents."""
    return {agent.get("case_id"): agent for agent in agents if "case_id" in agent}

def build_user_prompt(user_prompt_template, prompt_spec, agent1, agent2):
    """Render the user prompt for comparison using chevron and the template."""
    return chevron.render(user_prompt_template, {
        'prompt_spec': prompt_spec,
        'agent1_json': json.dumps(agent1, indent=2),
        'agent2_json': json.dumps(agent2, indent=2)
    })

def save_results_to_csv(dpo_results, csv_output_path):
    """Save DPO results to a CSV file."""
    df_results = pd.DataFrame({
        "case_id": [r["case_id"] for r in dpo_results],
        "base_prompt": [r["base_prompt"] for r in dpo_results],
        "agent_chosen": [r["agent_chosen"] for r in dpo_results],
        "agent_rejected": [r["agent_rejected"] for r in dpo_results],
        "justification": [r["justification"] for r in dpo_results]
    })
    df_results.to_csv(csv_output_path, index=False)
    logger.info(f"DPO dataset saved to {csv_output_path}")

def load_template(template_path):
    """Load a mustache template from file."""
    logging.info(f"Loading template from {template_path}")
    try:
        with open(template_path, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        logging.error(f"Error loading template: {e}")
        raise

    
    try:
        json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', judgment_text)
        
        if json_block_match:
            json_content = json_block_match.group(1)
            logger.info("Found JSON in code block")
        else:
            json_content = judgment_text
            logger.info("Using full text as JSON")
        
        json_content = json_content.strip()
        
        result = json.loads(json_content)
        
        preferred = result.get("choosen")
        rejected = result.get("rejected")
        justification = result.get("justification", "")
        
        logger.info(f"Successfully parsed JSON response with preferred={preferred}, rejected={rejected}")
        
        preferred_match = re.search(r'["\']{0,1}choosen["\']{0,1}\s*:\s*["\']{0,1}([AB])["\']{0,1}', judgment_text)
        rejected_match = re.search(r'["\']{0,1}rejected["\']{0,1}\s*:\s*["\']{0,1}([AB])["\']{0,1}', judgment_text)
        
        justification_match = re.search(r'["\']{0,1}justification["\']{0,1}\s*:\s*["\']{0,1}([\s\S]*?)["\']{0,1}(?:\s*[,}]|$)', judgment_text)
        
        preferred = preferred_match.group(1) if preferred_match else None
        rejected = rejected_match.group(1) if rejected_match else None
        justification = justification_match.group(1).strip() if justification_match else ""
        
        logger.info(f"Extracted using regex: preferred={preferred}, rejected={rejected}")
        
        return preferred, rejected, justification
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from judgment text: {e}")
        return None, None, ""

def main():
    """Main function to run the judge personality workflow."""
    load_dotenv()

    # Model settings
    provider = os.environ.get("MODEL_TYPE", "openai").lower()
    model_name = os.environ.get("MODEL_NAME", None)
    data_path = "dataset/train_set.csv" #os.environ.get("DATA_PATH")

    agent_path_exp1 = os.environ.get("AGENT_PATH_EXP1")
    agent_path_exp2 = os.environ.get("AGENT_PATH_EXP2")
    
    # Define prompt template paths (hardcoded)
    template_path = "src/templates/persona_template.mustache"
    system_prompt_path = "src/templates/judge_system_prompt.mustache"
    user_prompt_path = "src/templates/judge_user_prompt.mustache"
    
    # Load data
    logger.info("Loading training data...")
    data = pd.read_csv(data_path)
    
    # Initialize prompt builder
    logger.info("Initializing prompt builder...")
    prompt_builder = PromptBuilder(data, template_path)
    
    # Load agents data
    logger.info("Loading agent data...")
    agents_exp1 = read_json_file(agent_path_exp1)
    agents_exp2 = read_json_file(agent_path_exp2)
    # Create a mapping of case_id to agent data for easier lookup
    agents_exp1_map = map_agents_by_case_id(agents_exp1)
    agents_exp2_map = map_agents_by_case_id(agents_exp2)

    # Report on the data loading
    logger.info(f"Loaded {len(data)} data points, {len(agents_exp1)} agents from exp1, {len(agents_exp2)} agents from exp2")
    logger.info(f"Found {len(agents_exp1_map)} mappable agents in exp1, {len(agents_exp2_map)} mappable agents in exp2")

    
    # Prepare dataset for DPO (Direct Preference Optimization)
    dpo_results = []
    
    # Load prompt templates
    #logger.info("Loading prompt templates...")
    system_prompt_template = load_template(system_prompt_path)
    user_prompt_template = load_template(user_prompt_path)
    
    # Render system prompt (no variables needed)
    system_prompt = chevron.render(system_prompt_template, {})
    
    # Iterate through data samples
    logger.info("Starting evaluation process...")
    for idx in tqdm(range(len(data)), desc="Evaluating personalities"):
        # Get the case ID to match data rows with agents
        case_id = int(data.iloc[idx]["case"])

        # Find the corresponding agents using the maps
        agent1 = agents_exp1_map.get(case_id)
        agent2 = agents_exp2_map.get(case_id)
        
        if not agent1 or not agent2:
            continue
        
        # Build the original prompt
        prompt_spec = prompt_builder.build_prompt(idx)

        # Create user prompt for comparison by rendering the template
        user_prompt = build_user_prompt(user_prompt_template, prompt_spec, agent1, agent2)

        try:
            # Model Inference call using invoke_chat_model
            judgment = invoke_chat_model(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                provider=provider,
                model_name=model_name,
                extract_json=False
            )
            # Try to extract preferred, rejected, justification from model response
            try:
                result_json = extract_json_from_response(judgment)
                preferred = result_json.get("choosen")
                rejected = result_json.get("rejected")
                justification = result_json.get("justification", "")
            except Exception as e:
                logger.warning(f"Could not extract JSON for case {case_id}: {e}")
                preferred, rejected, justification = None, None, ""

            # Determine chosen and rejected agents
            if preferred == 'A':
                chosen_agent = agent1
                rejected_agent = agent2
            elif preferred == 'B':
                chosen_agent = agent2
                rejected_agent = agent1
            else:
                logger.warning(f"Invalid preference '{preferred}' from model for case {case_id}, skipping")
                continue

            # Record the result
            result = {
                "case_id": case_id,
                "base_prompt": prompt_spec,
                "agent_chosen": chosen_agent,
                "agent_rejected": rejected_agent,
                "justification": justification,
                "raw_judgment": judgment
            }

            dpo_results.append(result)

            # Save intermediate results periodically
            if len(dpo_results) % 10 == 0:
                intermediate_output_path = "dpo_dataset_intermediate.json"
                with open(intermediate_output_path, "w") as f:
                    json.dump(dpo_results, f, indent=2)
                logger.info(f"Saved {len(dpo_results)} intermediate results to {intermediate_output_path}")
        except Exception as e:
            logger.error(f"Error getting judgment for case {case_id}: {e}")

    # Save the DataFrame to CSV
    csv_output_path = "dpo_dataset.csv"
    save_results_to_csv(dpo_results, csv_output_path)


if __name__ == "__main__":
    main()