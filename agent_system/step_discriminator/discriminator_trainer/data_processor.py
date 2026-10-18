# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Data Processor for Step-Level Credit Discriminator Training.

This module reads credit analysis JSON files and generates Parquet training data
for fine-tuning a step-level credit discriminator model.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


# Prompt template for step classification (ALFWorld: includes observation)
PROMPT_TEMPLATE_WITH_OBS = """You are a trajectory analysis expert. Given a task and the complete execution trajectory, determine whether the specified step is necessary for completing the task.

## Task
{task}

## Complete Trajectory
{trajectory_steps}

## Step to Evaluate
Step {step_number}:
Action: {action}
Observation: {observation}

Is this step necessary for completing the task? Answer only "necessary" or "redundant"."""

# Prompt template for step classification (WebShop: trajectory is action-only, but step to evaluate includes observation)
PROMPT_TEMPLATE_ACTION_ONLY = """You are a trajectory analysis expert. Given a task and the complete execution trajectory, determine whether the specified step is necessary for completing the task.

## Task
{task}

## Complete Trajectory
{trajectory_steps}

## Step to Evaluate
Step {step_number}:
Action: {action}
Observation: {observation}

Is this step necessary for completing the task? Answer only "necessary" or "redundant"."""

# Keep PROMPT_TEMPLATE as alias for backwards compatibility (defaults to with observation)
PROMPT_TEMPLATE = PROMPT_TEMPLATE_WITH_OBS

# Valid labels for step classification
VALID_LABELS = {"necessary", "redundant"}

# Valid environment types
VALID_ENV_TYPES = {"alfworld", "webshop"}


def load_credit_analysis_file(input_file: str) -> List[Dict]:
    """
    Load a single credit analysis JSON file.
    
    Args:
        input_file: Path to the credit analysis JSON file
        
    Returns:
        List of analysis results from the file
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        results = data.get("analysis_results", [])
        print(f"  Loaded {len(results)} trajectory analysis results")
    
    return results


def build_trajectory_text(step_analysis: List[Dict], include_observation: bool = True) -> str:
    """
    Build trajectory text from step analysis list.
    
    Args:
        step_analysis: List of step analysis dictionaries
        include_observation: Whether to include observation in the text.
                           True for ALFWorld, False for WebShop.
        
    Returns:
        Formatted trajectory text
    """
    lines = []
    for step in step_analysis:
        step_num = step.get("step", 0)
        action = step.get("action", "")
        if include_observation:
            observation = step.get("observation", "")
            lines.append(f"Step {step_num}: Action: {action} | Observation: {observation}")
        else:
            lines.append(f"Step {step_num}: Action: {action}")
    return "\n".join(lines)


def build_prompt(task: str, step_analysis: List[Dict], target_step_idx: int, env_type: str = "alfworld") -> str:
    """
    Build classification prompt with full trajectory context.
    
    Args:
        task: Task description
        step_analysis: List of all step analysis dictionaries
        target_step_idx: Index of the step to evaluate
        env_type: Environment type, 'alfworld' or 'webshop'.
                 ALFWorld includes observation, WebShop is action-only.
        
    Returns:
        Formatted prompt string
    """
    include_observation = (env_type == "alfworld")
    trajectory_text = build_trajectory_text(step_analysis, include_observation=include_observation)
    target = step_analysis[target_step_idx]
    
    if include_observation:
        return PROMPT_TEMPLATE_WITH_OBS.format(
            task=task,
            trajectory_steps=trajectory_text,
            step_number=target.get("step", target_step_idx),
            action=target.get("action", ""),
            observation=target.get("observation", "")
        )
    else:
        # WebShop: trajectory is action-only, but step to evaluate includes observation
        return PROMPT_TEMPLATE_ACTION_ONLY.format(
            task=task,
            trajectory_steps=trajectory_text,
            step_number=target.get("step", target_step_idx),
            action=target.get("action", ""),
            observation=target.get("observation", "")
        )


def generate_samples(analysis_results: List[Dict], env_type: str = "alfworld") -> List[Dict]:
    """
    Generate training samples from analysis results.
    
    Each step with credit_level 'necessary' or 'redundant' becomes one sample.
    Steps with 'skipped_invalid' are filtered out.
    
    Args:
        analysis_results: List of trajectory analysis results
        env_type: Environment type, 'alfworld' or 'webshop'.
                 ALFWorld includes observation in prompt, WebShop is action-only.
        
    Returns:
        List of training samples with 'prompt' and 'response' keys
    """
    samples = []
    
    for traj_result in analysis_results:
        trajectory_info = traj_result.get("trajectory_info", {})
        task = trajectory_info.get("task", "")
        step_analysis = traj_result.get("step_analysis", [])
        
        if not task or not step_analysis:
            continue
        
        # Generate one sample for each valid step
        for idx, step in enumerate(step_analysis):
            credit_level = step.get("credit_level", "")
            
            # Skip invalid steps
            if credit_level == "skipped_invalid":
                continue
            
            # Validate credit level
            if credit_level not in VALID_LABELS:
                print(f"Warning: Unknown credit_level '{credit_level}', skipping")
                continue
            label = credit_level  # Use original value directly
            
            # Build prompt with env_type-specific format
            prompt = build_prompt(task, step_analysis, idx, env_type=env_type)
            
            samples.append({
                "prompt": prompt,
                "response": label,
            })
    
    return samples


def save_to_parquet(samples: List[Dict], output_path: str) -> None:
    """
    Save samples to Parquet file.
    
    Args:
        samples: List of sample dictionaries
        output_path: Path to output Parquet file
    """
    df = pd.DataFrame(samples)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(samples)} samples to {output_path}")


def process_data(
    input_file: str,
    output_dir: str,
    env_type: str = "alfworld",
) -> Dict:
    """
    Main data processing function.
    
    Args:
        input_file: Path to the credit analysis JSON file
        output_dir: Directory to save output Parquet files
        env_type: Environment type, 'alfworld' or 'webshop'.
                 ALFWorld includes observation in prompt, WebShop is action-only.
        
    Returns:
        Statistics dictionary
    """
    # Validate env_type
    if env_type not in VALID_ENV_TYPES:
        raise ValueError(f"Invalid env_type '{env_type}'. Must be one of: {VALID_ENV_TYPES}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load analysis file
    print("=" * 60)
    print("Loading credit analysis file...")
    analysis_results = load_credit_analysis_file(input_file)
    
    # Generate samples with env_type-specific format
    print("=" * 60)
    print(f"Generating training samples (env_type={env_type})...")
    samples = generate_samples(analysis_results, env_type=env_type)
    print(f"Generated {len(samples)} total samples")
    
    # Count label distribution
    necessary_count = sum(1 for s in samples if s["response"] == "necessary")
    redundant_count = sum(1 for s in samples if s["response"] == "redundant")
    print(f"  - necessary: {necessary_count}")
    print(f"  - redundant: {redundant_count}")
    
    # Save to Parquet
    print("=" * 60)
    print("Saving to Parquet file...")
    train_path = os.path.join(output_dir, "train.parquet")
    save_to_parquet(samples, train_path)
    
    # Return statistics
    stats = {
        "env_type": env_type,
        "total_trajectories": len(analysis_results),
        "total_samples": len(samples),
        "necessary_samples": necessary_count,
        "redundant_samples": redundant_count,
        "train_path": train_path,
    }
    
    print("=" * 60)
    print("Data processing complete!")
    print(f"Statistics: {json.dumps(stats, indent=2)}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Process credit analysis data for discriminator training"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the credit analysis JSON file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output Parquet files"
    )
    parser.add_argument(
        "--env_type",
        type=str,
        default="alfworld",
        choices=["alfworld", "webshop"],
        help="Environment type: 'alfworld' (includes observation) or 'webshop' (action-only). Default: alfworld"
    )
    
    args = parser.parse_args()
    
    process_data(
        input_file=args.input_file,
        output_dir=args.output_dir,
        env_type=args.env_type,
    )


if __name__ == "__main__":
    main()
