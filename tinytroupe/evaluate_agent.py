import os
import json
import glob
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from tinytroupe.agent.tiny_person import TinyPerson
from tinytroupe.extraction import ResultsExtractor

# Constants
LETTER_TO_INTEGER = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5
}

QUESTION_TEMPLATE = """Given a statement of you: "You {}."
You must choose from the following options to identify how accurately this statement describes you.
Options:
(A). Very Accurate
(B). Moderately Accurate
(C). Neither Accurate Nor Inaccurate
(D). Moderately Inaccurate
(E). Very Inaccurate
"""

# Configuration functions
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate agent responses to IPIP-120 statements.')
    parser.add_argument('--agent-path', type=str, default="tinytroupe/examples/evaluation_ipip_120/agents/test_set/trained_qwen/all_agents_test.json",
                        help='Path to the agent JSON file')
    parser.add_argument('--data-path', type=str, default="tinytroupe/examples/evaluation_ipip_120/data/mpi_120.csv",
                        help='Path to the IPIP-120 CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for output files (defaults to script directory)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of questions to process')
    parser.add_argument('--checkpoint-every', type=int, default=1,
                        help='Save intermediate results every N questions')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    return parser.parse_args()

def setup_logging(log_level):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_agent_data(agent_path):
    """Load agent data from JSON file."""
    logging.info(f"Loading agent data from {agent_path}")
    try:
        with open(agent_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading agent data: {e}")
        raise

def load_questionnaire_data(data_path, limit=None):
    """Load IPIP-120 statements from CSV file."""
    logging.info(f"Loading questionnaire data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        return df
    except Exception as e:
        logging.error(f"Error loading questionnaire data: {e}")
        raise

def collect_processed_case_ids(output_dir):
    """Collect case IDs that have already been processed."""
    file = "tinytroupe/examples/evaluation_ipip_120/mpi_120_results_batch_1.csv"
    processed_case_ids = set()

    try:
        intermediate_df = pd.read_csv(file)
        processed_case_ids.update(intermediate_df["case"].tolist())
    except Exception as e:
        logging.warning(f"Error reading {file}: {e}")
    
    logging.info(f"Found {len(processed_case_ids)} previously processed case IDs")
    return processed_case_ids

def prepare_agent_structure(agent_info):
    """Ensure agent info has the correct structure."""
    if not ("type" in agent_info and "persona" in agent_info):
        logging.debug("Adding outer structure to agent info")
        return {
            "type": "TinyPerson",
            "persona": agent_info
        }
    return agent_info

def initialize_agent(agent_info):
    """Initialize an agent from its specification."""
    try:
        return TinyPerson.load_specification(agent_info)
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        raise

def extract_agent_answer(agent):
    """Extract the agent's answer to a question."""
    extractor = ResultsExtractor()
    extraction_objective = """Find only of the following options from the last ACTION:
                            A
                            B
                            C
                            D
                            E
                        """
    try:
        res = extractor.extract_results_from_agent(
            agent,
            extraction_objective=extraction_objective,
            situation="",
            fields=["option"],
            fields_hints={"option": "Must be a letter A,B,C,D,E."},
            verbose=True
        )
        return res
    except Exception as e:
        logging.error(f"Error extracting agent answer: {e}")
        return {}

def extract_option_value(ans):
    """Convert the agent's answer to a numeric value."""
    if isinstance(ans, dict):
        option = ans.get('option', '')
        value = LETTER_TO_INTEGER.get(option, 0)
        logging.info(f"Extracted option: {option}, numeric value: {value}")
        if value == 0:
            logging.warning(f"Invalid option value: {option}")
        return value
    else:
        logging.warning(f"Unexpected format for answer: {type(ans)}")
        return 0

def extract_assistant_action_content(memories):
    """Extract the integer value associated with the assistant's action content from the memories list."""
    # Define mapping of statements to letters
    statement_to_letter = {
        "Very Accurate": "A",
        "Moderately Accurate": "B",
        "Neither Accurate Nor Inaccurate": "C",
        "Moderately Inaccurate": "D",
        "Very Inaccurate": "E"
    }

    for memory in memories:
        if memory.get('role') == 'assistant':
            action = memory.get('content', {}).get('action', {})
            content = action.get('content', None)
            if content:
                # Check if content contains a letter in the format "(X)."
                import re
                match = re.search(r"\((A|B|C|D|E)\)\.", content)
                if match:
                    letter = match.group(1)  # Extract the letter inside parentheses
                    return LETTER_TO_INTEGER.get(letter, 0)  # Map letter to integer
                # Check if content matches a statement
                for statement, letter in statement_to_letter.items():
                    if statement in content:
                        return LETTER_TO_INTEGER.get(letter, 0)  # Map letter to integer
    return 0

def process_question(agent, statement):
    """Process a single question with the agent."""
    question = QUESTION_TEMPLATE.format(statement)
    agent.listen_and_act(question, n=1)
    memories = agent.retrieve_memories(first_n=None, last_n=None)
    logging.info(f"Agent memories after question: {memories}, type: {type(memories)}")
    ans = extract_assistant_action_content(memories)
    logging.info(f"Extracted assistant action content: {ans}")
    #ans = extract_agent_answer(agent)
    # reset memory here or internally?
    agent.manage_memory(reset=True)
    return ans

def save_intermediate_results(results, output_dir, step):
    """Save intermediate results to a CSV file."""
    if not results:
        logging.info("No results to save")
        return

    intermediate_results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, f"mpi_120_results_step_{step}.csv")

    try:
        intermediate_results_df.to_csv(output_path, index=False)
        logging.info(f"Intermediate results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving intermediate results: {e}")

def save_final_results(results, output_dir):
    """Save final results to a CSV file."""
    if not results:
        logging.warning("No results to save")
        return
    
    results_df = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "mpi_120_results.csv")
    
    try:
        results_df.to_csv(output_path, index=False)
        logging.info(f"Final results saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving final results: {e}")

def process_agent(agent_info, questionnaire_df, processed_case_ids, output_dir, checkpoint_every):
    """Process a single agent with all questions."""
    case_id = agent_info.get("case_id", None)
    
    # Skip if already processed
    if case_id in processed_case_ids:
        logging.info(f"Skipping agent with case_id: {case_id} (already processed)")
        return None
    
    logging.info(f"Processing agent with case_id: {case_id}")
    
    # Prepare agent
    prepared_agent_info = prepare_agent_structure(agent_info)
    agent = initialize_agent(prepared_agent_info)
    
    # Initialize results
    agent_results = {"case": case_id}
    
    # Process each question with tqdm progress bar
    for idx, row in tqdm(list(questionnaire_df.iterrows()), total=len(questionnaire_df), desc=f"Questions for agent {case_id}"):
        statement = row["text"]
        logging.debug(f"Processing statement {idx+1}: {statement}")
        numeric_answer = process_question(agent, statement)
        agent_results[f"i{idx + 1}"] = numeric_answer
    
    # Store agent's memory
    agent_results["memory"] = agent.retrieve_memories(first_n=None, last_n=None)
    return agent_results

def main():
    """Main execution flow."""
    # Parse arguments and setup
    args = parse_arguments()
    setup_logging(args.log_level)
    
    # Set output directory
    output_dir = args.output_dir or os.path.dirname(__file__)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    agent_data = load_agent_data(args.agent_path)
    logging.info(f"Loaded {len(agent_data)} agents from {args.agent_path}")
    questionnaire_df = load_questionnaire_data(args.data_path, None)
    processed_case_ids = [] #collect_processed_case_ids(output_dir)
    
    # Optionally limit the number of agents
    if args.limit is not None:
        agent_data_limited = agent_data[:args.limit]
    else:
        agent_data_limited = agent_data

    # Process all agents with tqdm progress bar
    results = []
    for agent_idx, agent_info in enumerate(tqdm(agent_data_limited, desc="Agents")):
        agent_result = process_agent(
            agent_info, 
            questionnaire_df, 
            processed_case_ids, 
            output_dir, 
            args.checkpoint_every
        )
        if agent_result:
            results.append(agent_result)

    # Save final results
    save_final_results(results, output_dir)

if __name__ == "__main__":
    main()