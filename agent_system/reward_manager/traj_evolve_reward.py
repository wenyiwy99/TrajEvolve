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
TrajEvolve Reward Manager.

This module implements the reward computation for TrajEvolve algorithm:
- Success trajectories: base + alpha * efficiency + beta * efficiency^n / necessary_steps
  - When redundant steps exist: efficiency term dominates (encourages high efficiency)
  - When efficiency = 1.0: brevity term activates (encourages fewer steps)
- Failed trajectories: reward = 0 (no penalty, unchanged from environment)
"""

from verl import DataProto
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple


class TrajEvolveRewardManager:
    """
    Reward manager for TrajEvolve algorithm.
    
    For success trajectories:
        reward = base - alpha * redundancy_ratio
        where redundancy_ratio = 1 - necessary_steps / total_steps
        
        This formula penalizes redundant steps:
        - When necessary_steps = total_steps: redundancy_ratio = 0, no penalty
        - When redundant steps exist: redundancy_ratio > 0, reward decreases
    
    For failed trajectories:
        reward = 0 (unchanged from environment, no additional penalty)
    """

    def __init__(
        self,
        tokenizer,
        num_examine: int = 0,
        normalize_by_length: bool = False,
        alpha: float = 1.0,
    ) -> None:
        """
        Initialize TrajEvolve reward manager.
        
        Args:
            tokenizer: Tokenizer for decoding responses
            num_examine: Number of batches to print for debugging
            normalize_by_length: Whether to normalize reward by episode length
            alpha: Weight for redundancy penalty term (default 1.0)
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.normalize_by_length = normalize_by_length
        self.alpha = alpha

    def compute_success_trajectory_reward(
        self,
        episode_reward: float,
        necessary_steps: int,
        total_steps: int,
    ) -> float:
        """
        Compute reward for a successful trajectory.
        
        Formula: reward = base - alpha * redundancy_ratio
        where redundancy_ratio = 1 - necessary_steps / total_steps
        
        Args:
            episode_reward: Original episode reward (typically 10.0 for ALFWorld success)
            necessary_steps: Number of necessary (non-redundant) steps
            total_steps: Total number of steps in trajectory
            
        Returns:
            Adjusted reward with redundancy penalty
        """
        if total_steps <= 0 or necessary_steps <= 0:
            return episode_reward
        
        # Redundancy ratio: 1 - necessary / total (0~1)
        # 0 means no redundant steps, 1 means all steps are redundant
        redundancy_ratio = 1 - necessary_steps / total_steps
        
        # Redundancy penalty: alpha * redundancy_ratio
        redundancy_penalty = self.alpha * redundancy_ratio
        
        # reward = base - alpha * redundancy_ratio
        return episode_reward - redundancy_penalty

    def __call__(
        self,
        data: DataProto,
        success_labels: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ):
        """
        Compute rewards for a batch of trajectories.
        
        Args:
            data: DataProto containing trajectory data
            success_labels: Dict mapping traj_uid to credit analysis results
                           {'traj_uid': {'necessary_count': int, 'total_steps': int}}
            
        Returns:
            Dict with reward_tensor and extra info
        """
        # If there is rm score, directly return rm score
        if "rm_scores" in data.batch.keys():
            return {"reward_tensor": data.batch["rm_scores"]}

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()

            episode_rewards = data_item.non_tensor_batch['episode_rewards']
            episode_lengths = data_item.non_tensor_batch['episode_lengths']
            traj_uid = data_item.non_tensor_batch.get('traj_uid', None)
            
            # Determine if this is a success or failed trajectory
            is_success = episode_rewards > 0.5  # ALFWorld: success=10.0, failure=0.0
            
            if is_success and success_labels is not None and traj_uid in success_labels:
                # Success trajectory: use efficiency + brevity based reward
                label_info = success_labels[traj_uid]
                necessary_count = label_info.get('necessary_count', int(episode_lengths))
                total_steps = label_info.get('total_steps', int(episode_lengths))
                
                score = self.compute_success_trajectory_reward(
                    episode_reward=episode_rewards,
                    necessary_steps=necessary_count,
                    total_steps=total_steps,
                )
                
                if self.normalize_by_length and episode_lengths > 0:
                    score = score / episode_lengths
            else:
                # Fallback: use original episode reward
                if self.normalize_by_length and episode_lengths > 0:
                    score = episode_rewards / episode_lengths
                else:
                    score = episode_rewards

            # Assign reward to the last valid token position
            reward_tensor[i, valid_response_length - 1] = torch.tensor(
                score, dtype=torch.float32, device=prompt_ids.device
            )

            # Debug printing
            data_source = data_item.non_tensor_batch.get('data_source', 'unknown')
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine and np.random.random() < 0.1:
                already_print_data_sources[data_source] += 1
                valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                response_ids = data_item.batch['responses']
                valid_response_ids = response_ids[:valid_response_length]
                
                prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
                response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False)
                
                print(f"[{data_source}][TrajEvolve][prompt]", prompt_str[:200], "...")
                print(f"[{data_source}][TrajEvolve][response]", response_str[:200], "...")
                print(f"[{data_source}][TrajEvolve][score]", score)
                print(f"[{data_source}][TrajEvolve][is_success]", is_success)

        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": {},
        }


def compute_trajectory_efficiency(analysis_result: Dict) -> Tuple[int, int]:
    """
    Extract necessary steps count and total steps from credit analysis result.
    
    Reads from the 'step_analysis' list produced by CreditLevelChecker.to_dict().
    
    Args:
        analysis_result: Result from analyze_trajectory_credit_level(),
            must contain 'step_analysis' list with 'credit_level' per step.
        
    Returns:
        Tuple of (necessary_count, total_steps)
        
    Raises:
        ValueError: If 'step_analysis' is missing or empty in analysis_result.
    """
    step_analysis = analysis_result.get('step_analysis')
    if not step_analysis:
        raise ValueError(
            f"analysis_result missing 'step_analysis' field. "
            f"Available keys: {list(analysis_result.keys())}"
        )
    
    necessary_count = sum(1 for s in step_analysis if s.get('credit_level') == 'necessary')
    total_steps = len(step_analysis)
    
    return necessary_count, total_steps


def prepare_success_labels_from_analysis(
    analysis_results: List[Dict],
    trajectories: List[Dict],
) -> Dict[str, Dict]:
    """
    Prepare success trajectory labels from credit analysis results.
    
    Args:
        analysis_results: List of credit analysis results
        trajectories: List of trajectory dicts with 'traj_uid' key
        
    Returns:
        Dict mapping traj_uid to label info
    """
    labels = {}
    
    for traj, analysis in zip(trajectories, analysis_results):
        traj_uid = traj.get('traj_uid')
        if traj_uid is None:
            continue
            
        necessary_count, total_steps = compute_trajectory_efficiency(analysis)
        
        labels[traj_uid] = {
            'necessary_count': necessary_count,
            'total_steps': total_steps,
            'analysis_result': analysis,
        }
    
    return labels
