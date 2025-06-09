# Project Overview

This repository implements a pipeline for generating, evaluating, and training personality-based agent models using the IPIP-NEO-120 personality inventory. The workflow consists of several key processes, each handled by a dedicated script or module:

## 1. process_dataset
- **Script:** `process_dataset.py`
- **Purpose:** Processes raw personality assessment data to produce clean, structured datasets for downstream tasks.
- **Workflow:**
  - Loads raw IPIP-NEO-120 data and mapping files.
  - Computes facet and domain scores for each individual.
  - Adds demographic group classifications.
  - Calculates percentiles for demographic groups and categorizes scores accordingly.
  - Outputs processed datasets for use in agent generation and evaluation.

## 2. create_agent_spec
- **Script:** `create_agent_spec.py`
- **Purpose:** Generates detailed persona specifications (agent profiles) based on processed personality trait data.
- **Workflow:**
  - Loads processed data and a prompt template.
  - Builds prompts and uses language models (e.g., OpenAI, Together AI) to generate agent specifications.
  - Supports sampling, inclusion/exclusion of specific cases, and logging.
  - Outputs agent specification files for use in evaluation and training.

## 3. judge_personality
- **Script:** `judge_agent_spec.py` (sometimes referred to as `judge_personality`)
- **Purpose:** Compares pairs of agent specifications to generate preference data for Direct Preference Optimization (DPO) training.
- **Workflow:**
  - Loads two sets of agent specifications and the relevant prompt templates.
  - For each case, builds a comparison prompt and queries a language model to judge which agent better matches the prompt specification.
  - Extracts the model's preference and justification.
  - Outputs a dataset of preferences and justifications for use in DPO model training.

## 4. model training
- **Purpose:** Fine-tunes or trains models using the preference data generated in the previous step (DPO dataset).
- **Workflow:**
  - Uses the DPO dataset (e.g., `dpo_dataset.csv`) to train or fine-tune a model to better align with human or model-based preferences.
  - The specifics of the training process (framework, scripts, etc.) are not detailed in the provided files, but this step is a standard ML training loop using the generated data.

## 5. run_evaluation
- **Script:** `run_evaluation.py`
- **Purpose:** Evaluates the performance of baseline and trained models on personality assessment tasks.
- **Workflow:**
  - Loads baseline and processed results, as well as percentile data.
  - Computes and compares scores, agreement, and accuracy at various levels (question, facet, domain, agent).
  - Outputs comparative metrics and visualizations (e.g., heatmaps) to assess model performance.

---


---

## Using the TinyTroupe Framework

This repository integrates the TinyTroupe framework to simulate agent responses to the IPIP-120 personality test. The script `tinytroupe/evaluate_agent.py` demonstrates how to:

- Load agent specifications (personas) generated in earlier steps.
- Use TinyTroupe's `TinyPerson` class to instantiate agents with these personas.
- Present each IPIP-120 statement to the agent and collect its answer using TinyTroupe's action and memory management APIs.
- Extract and record the agent's responses in a structured format for further analysis or evaluation.

This implementation serves as a practical example of how to consume TinyTroupe's functionalities for automated, reproducible agent-based evaluation on standardized psychological assessments.
