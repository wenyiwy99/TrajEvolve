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


import os
import glob
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pprint import pprint

import ray
import torch
from tqdm import tqdm
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage, apply_kl_penalty, apply_invalid_action_penalty, compute_response_mask, _timer, AdvantageEstimator
from verl.utils.model import compute_position_id_with_mask
from gigpo import core_gigpo

from agent_system.step_discriminator import (
    analyze_trajectory_credit_level,
)
# from agent_system.environments.prompts.alfworld import ALFWORLD_TEMPLATE
# from agent_system.environments.prompts.webshop import WEBSHOP_TEMPLATE
# from agent_system.environments.env_package.alfworld import map_to_admissible_alfworld
from agent_system.step_discriminator.discriminator_trainer.data_processor import generate_samples, save_to_parquet
from agent_system.reward_manager import (
    TrajEvolveRewardManager,
    prepare_success_labels_from_analysis,
)


def map_to_admissible_webshop(action: str, available_actions: dict) -> str:
    """Map a malformed WebShop click action to the closest clickable via longest common substring."""
    def find_longest_common_substring(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        if m == 0 or n == 0:
            return 0
        dp = [[0] * (n + 1) for _ in range(2)]
        max_len = 0
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i % 2][j] = dp[(i-1) % 2][j-1] + 1
                    max_len = max(max_len, dp[i % 2][j])
                else:
                    dp[i % 2][j] = 0
        return max_len
    
    clickables = available_actions.get('clickables', [])
    if not clickables:
        return action
    
    action_lower = action.lower()
    
    # Try to extract content from click[xxx]
    click_content = action_lower
    if 'click[' in action_lower and ']' in action_lower:
        click_start = action_lower.find('click[') + len('click[')
        click_end = action_lower.rfind(']')
        if click_end > click_start:
            click_content = action_lower[click_start:click_end].strip()
    
    # Exact substring match first
    for clickable in clickables:
        if clickable.lower() in click_content:
            return f"click[{clickable}]"
    
    # Fallback: longest common substring match
    best_clickable = clickables[0]
    best_score = 0
    for clickable in clickables:
        score = find_longest_common_substring(click_content, clickable.lower())
        if score > best_score:
            best_score = score
            best_clickable = clickable
    
    return f"click[{best_clickable}]"


def find_first_format_error(trajectory: List[Dict]) -> Optional[int]:
    """Return the index of the first is_action_valid=False step (ALFWorld), or None."""
    for i, step in enumerate(trajectory):
        if not step.get('is_action_valid', True):
            return i
    return None


def find_first_invalid_action_webshop(trajectory: List[Dict]) -> tuple:
    """Find the first invalid action in a WebShop trajectory and classify error type.

    Returns:
        (index, error_type): error_type is 'format_error' (contains 'cli') or 'action_error'.
        (None, None) if no invalid action found.
    """
    for i, step in enumerate(trajectory):
        if not step.get('is_action_valid', True):
            action_text = step.get('action_text', '').lower()
            if 'cli' in action_text:
                return i, 'format_error'
            else:
                return i, 'action_error'
    return None, None


@ray.remote
def _analyze_single_trajectory_remote(traj: Dict, env_type: str = 'alfworld') -> Optional[Dict]:
    """Ray remote wrapper for parallel trajectory credit analysis."""
    try:
        return analyze_trajectory_credit_level(traj, env_type=env_type)
    except Exception as e:
        print(f"[TrajEvolve] Error analyzing trajectory: {e}")
        return None


class TrajEvolveTrainer(RayPPOTrainer):
    """Extends RayPPOTrainer with trajectory evolution capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

        if "webshop" in self.config.env.env_name.lower():
            self.env_type = 'webshop'
        else:
            self.env_type = 'alfworld'
        
        self.traj_evolve_config = self.config.get('traj_evolve', {})
        
        # Reward formula: base - alpha * redundancy_ratio
        self.traj_evolve_reward_fn = TrajEvolveRewardManager(
            tokenizer=self.tokenizer,
            num_examine=0,
            normalize_by_length=False,
            alpha=self.traj_evolve_config.get('alpha', 1.0),
        )
        
        self.discriminator_model_path = self.traj_evolve_config.get('discriminator_model_path')
        self.discriminator_output_dir = self.traj_evolve_config.get('discriminator_output_dir', '/tmp/discriminator_output')
        self.evolve_freq = self.traj_evolve_config.get('evolve_freq', 5)
        
        # Accumulated storage across evolve_freq epochs
        self.accumulated_trajectories = []
        self.accumulated_analysis_results = []
        self.accumulated_failed_labels = {}
        
        self.step_credit_value = self.traj_evolve_config.get('step_credit_value', 0.5)
        
        # rewards_cache: {group_key: [reward1, reward2, ...]}
        self.rewards_cache = {}
        
        print(f"[TrajEvolve] Enabled with alpha={self.traj_evolve_config.get('alpha', 0.1)}, step_credit_value={self.step_credit_value}")
        
        self.discriminator_wg = None
    
    def init_workers(self):
        """Initialize all workers including the colocated DiscriminatorSFT worker."""
        from verl.trainer.ppo.ray_trainer import Role, RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls
        from omegaconf import OmegaConf
        
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError
        
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        
        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        
        # DiscriminatorSFT: colocated with other workers
        if Role.DiscriminatorSFT in self.role_worker_mapping:
            discriminator_config = self._create_discriminator_config()
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.DiscriminatorSFT)
            discriminator_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.DiscriminatorSFT], 
                config=discriminator_config
            )
            self.resource_pool_to_cls[resource_pool]["discriminator_sft"] = discriminator_cls
            print("[TrajEvolve] DiscriminatorSFT added to resource pool")
        
        # Create colocated worker groups
        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, 
                ray_cls_with_init=worker_dict_cls, 
                device_name=self.device_name, 
                **wg_kwargs
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()
        
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # Init actor_rollout last for better vllm kv cache estimation
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
        if "discriminator_sft" in all_wg:
            self.discriminator_wg = all_wg["discriminator_sft"]
            self.discriminator_wg.init_model()
            print("[TrajEvolve] Discriminator SFT Worker initialized successfully")
        
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            from verl.trainer.ppo.ray_trainer import AsyncLLMServerManager
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )
    
    def _create_discriminator_config(self):
        """Create OmegaConf config for DiscriminatorSFTWorker."""
        from omegaconf import OmegaConf
        
        base_model_path = self.discriminator_model_path or self.config.actor_rollout_ref.model.path
        
        use_lora = self.traj_evolve_config.get('discriminator_use_lora', True)
        lora_rank = self.traj_evolve_config.get('discriminator_lora_rank', 64) if use_lora else 0
        lora_alpha = self.traj_evolve_config.get('discriminator_lora_alpha', 128) if use_lora else 0
        
        discriminator_config = OmegaConf.create({
            'model': {
                'path': base_model_path,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'target_modules': ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                'trust_remote_code': self.config.actor_rollout_ref.model.get('trust_remote_code', False),
                'enable_gradient_checkpointing': True,
                'use_remove_padding': False,
                'fsdp_config': {
                    'fsdp_size': -1,
                    'param_offload': False,
                    'optimizer_offload': False,
                    'model_dtype': 'bf16',
                },
            },
            'optim': {
                'lr': self.traj_evolve_config.get('discriminator_lr', 1e-5),
                'betas': [0.9, 0.999],
                'weight_decay': 0.01,
                'warmup_style': 'cosine',
                'lr_warmup_steps_ratio': 0.1,
                'total_training_steps': 1000,
            },
            'strategy': self.config.actor_rollout_ref.actor.strategy,
            'ulysses_sequence_parallel_size': 1,
        })
        
        return discriminator_config
    
    
    def _analyze_successful_trajectories(
        self,
        trajectories: List[Dict],
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        """Analyze successful trajectories via counterfactual verification (Ray parallel).

        Returns:
            (analysis_results, success_labels) for reward computation.
        """
        successful_trajs = [t for t in trajectories if t.get('is_success', False)]
        
        if not successful_trajs:
            print("[TrajEvolve] No successful trajectories to analyze")
            return [], {}
        
        print(f"[TrajEvolve] Analyzing {len(successful_trajs)} trajectories (Ray parallel)...")
        
        futures = [_analyze_single_trajectory_remote.remote(traj, self.env_type) for traj in successful_trajs]
        results = ray.get(futures)
        
        analysis_results = []
        valid_trajs = []
        for traj, result in zip(successful_trajs, results):
            if result is not None:
                analysis_results.append(result)
                valid_trajs.append(traj)
        
        success_labels = prepare_success_labels_from_analysis(
            analysis_results=analysis_results,
            trajectories=valid_trajs,
        )
        
        print(f"[TrajEvolve] Analyzed {len(analysis_results)} trajectories")
        return analysis_results, success_labels
    
    def _train_discriminator(
        self,
        analysis_results: List[Dict],
        epoch: int,
    ) -> None:
        """Train discriminator on credit analysis results. Model is updated in-place on GPU."""
        if not analysis_results:
            print("[TrajEvolve] No analysis results for discriminator training")
            return
        
        if self.discriminator_wg is None:
            print("[TrajEvolve] Warning: discriminator_wg not initialized, skipping discriminator training")
            return
        
        epoch_output_dir = os.path.join(self.discriminator_output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        
        samples = generate_samples(analysis_results, env_type=self.env_type)
        
        min_samples_for_training = self.traj_evolve_config.get('discriminator_min_samples', 16)
        if len(samples) < min_samples_for_training:
            print(f"[TrajEvolve] Skipping discriminator training: only {len(samples)} samples (need >= {min_samples_for_training})")
            return
        
        train_parquet_path = os.path.join(epoch_output_dir, "train.parquet")
        save_to_parquet(samples, train_parquet_path)
        
        print(f"[TrajEvolve] Training discriminator with {len(samples)} samples...")
        
        discriminator_lr = self.traj_evolve_config.get('discriminator_lr', 1e-5)
        discriminator_epochs = self.traj_evolve_config.get('discriminator_epochs', 1)
        discriminator_max_batch_size = self.traj_evolve_config.get('discriminator_micro_batch_size', 8)
        discriminator_max_length = self.traj_evolve_config.get('discriminator_max_length', 4096)
        
        # Dynamic micro_batch_size: ensure world_size * micro_batch_size <= len(samples)
        world_size = self.traj_evolve_config.get('world_size', 8)
        samples_per_rank = len(samples) // world_size
        discriminator_micro_batch_size = max(1, min(discriminator_max_batch_size, samples_per_rank))
        
        print(f"[TrajEvolve] Dynamic batch size: {discriminator_micro_batch_size} "
              f"(samples={len(samples)}, world_size={world_size}, max={discriminator_max_batch_size})")
        
        train_config = {
            'lr': discriminator_lr,
            'epochs': discriminator_epochs,
            'micro_batch_size': discriminator_micro_batch_size,
            'max_length': discriminator_max_length,
        }
        
        try:
            print(f"[TrajEvolve] Starting Discriminator SFT training via discriminator_wg...")
            self.discriminator_wg.train_sft(
                train_data_path=train_parquet_path,
                train_config=train_config
            )
            print(f"[TrajEvolve] Discriminator training completed (model updated in GPU memory)")
        except Exception as e:
            print(f"[TrajEvolve] Discriminator training error: {e}")
            import traceback
            traceback.print_exc()
    
    def _label_failed_trajectories(
        self,
        trajectories: List[Dict],
    ) -> Dict[str, Dict]:
        """Label failed trajectories with first error step and error type using the discriminator.

        Checks format errors first, then uses model inference for logic errors.

        Returns:
            {traj_uid: {'first_error_step': int, 'error_type': 'format_error'|'action_error'}}
        """
        from agent_system.step_discriminator.discriminator_trainer.data_processor import build_prompt
        
        failed_trajs = [t for t in trajectories if not t.get('is_success', False)]
        
        if not failed_trajs:
            print("[TrajEvolve] No failed trajectories to label")
            return {}
        
        if self.discriminator_wg is None:
            print("[TrajEvolve] Warning: discriminator_wg not initialized, using default labels")
            return {
                t.get('traj_uid'): {
                    'first_error_step': 0,
                    'error_type': 'action_error'
                } 
                for t in failed_trajs if t.get('traj_uid')
            }
        
        print(f"[TrajEvolve] Labeling {len(failed_trajs)} selected failed trajectories via discriminator_wg...")
        
        try:
            results = {}
            
            # Preprocess: find format errors and collect prompts for batch inference
            traj_meta = {}
            all_prompts = []
            prompt_map = []  # (traj_uid, step_idx) for each prompt
            
            for traj in failed_trajs:
                traj_uid = traj.get('traj_uid')
                task = traj.get('task', '')
                trajectory = traj.get('trajectory', [])
                
                if not traj_uid or not trajectory:
                    continue
                
                # Find first invalid action (env-specific logic)
                if self.env_type == 'webshop':
                    invalid_action_idx, invalid_action_error_type = find_first_invalid_action_webshop(trajectory)
                    check_until = invalid_action_idx if invalid_action_idx is not None else len(trajectory)
                else:
                    invalid_action_idx = find_first_format_error(trajectory)
                    invalid_action_error_type = 'format_error' if invalid_action_idx is not None else None
                    check_until = invalid_action_idx if invalid_action_idx is not None else len(trajectory)
                
                traj_meta[traj_uid] = {
                    'task': task,
                    'trajectory': trajectory,
                    'invalid_action_idx': invalid_action_idx,
                    'invalid_action_error_type': invalid_action_error_type,
                    'check_until': check_until,
                }
                
                # Build prompts for steps before the invalid action
                step_analysis = []
                for i, step in enumerate(trajectory):
                    step_analysis.append({
                        'step': i,
                        'action': step.get('action_text', step.get('action', '')),
                        'observation': step.get('observation_text', step.get('observation', '')),
                    })
                
                for step_idx in range(check_until):
                    prompt = build_prompt(task, step_analysis, step_idx, env_type=self.env_type)
                    all_prompts.append(prompt)
                    prompt_map.append((traj_uid, step_idx))
            
            if not traj_meta:
                return {}
            
            print(f"[TrajEvolve] Batch inference: {len(all_prompts)} prompts from {len(traj_meta)} trajectories")
            
            # Batch inference
            if all_prompts:
                predictions = self.discriminator_wg.predict_batch(prompts=all_prompts, batch_size=8)
                if isinstance(predictions, list) and len(predictions) > 0:
                    if isinstance(predictions[0], list):
                        predictions = predictions[0]  # use rank 0's result
            else:
                predictions = []
            
            # Organize predictions by trajectory
            traj_predictions = {uid: {} for uid in traj_meta}
            for (traj_uid, step_idx), pred in zip(prompt_map, predictions):
                traj_predictions[traj_uid][step_idx] = pred
            
            # Determine first error for each trajectory
            for traj_uid, meta in traj_meta.items():
                invalid_action_idx = meta['invalid_action_idx']
                invalid_action_error_type = meta['invalid_action_error_type']
                check_until = meta['check_until']
                trajectory = meta['trajectory']
                step_preds = traj_predictions[traj_uid]
                
                # Discriminator-detected logic error takes priority
                first_discriminator_action_error = None
                for step_idx in range(check_until):
                    if step_preds.get(step_idx) == "redundant":
                        first_discriminator_action_error = step_idx
                        break
                
                if first_discriminator_action_error is not None:
                    results[traj_uid] = {'first_error_step': first_discriminator_action_error, 'error_type': 'action_error'}
                elif invalid_action_idx is not None:
                    results[traj_uid] = {'first_error_step': invalid_action_idx, 'error_type': invalid_action_error_type}
                else:
                    results[traj_uid] = {'first_error_step': len(trajectory) - 1, 'error_type': 'action_error'}
            
            print(f"[TrajEvolve] Labeled {len(results)} failed trajectories")
            return results
            
        except Exception as e:
            print(f"[TrajEvolve] Error labeling failed trajectories: {e}")
            import traceback
            traceback.print_exc()
            return {
                t.get('traj_uid'): {
                    'first_error_step': len(t.get('trajectory', [])) - 1,
                    'error_type': 'action_error'
                } 
                for t in failed_trajs if t.get('traj_uid')
            }
    
    def _apply_step_credit_adjustment(
        self,
        batch: DataProto,
        analysis_results: List[Dict],
        failed_labels: Dict[str, Dict],
        step_credit_value: float = 0.5,
    ) -> DataProto:
        """Adjust advantages with step-level credit after normalization.

        - Success necessary steps: +step_credit_value
        - Failed steps before first_error: +step_credit_value
        - Failed first_error step: -step_credit_value
        """
        if step_credit_value <= 0:
            return batch
        
        # Build traj_uid -> {step_idx: credit_level} map for success trajectories
        success_credit_map = {}
        for result in analysis_results:
            traj_info = result.get('trajectory_info', {})
            traj_uid = traj_info.get('traj_uid')
            if traj_uid is None:
                continue
            
            step_credits = {}
            for step in result.get('step_analysis', []):
                step_credits[step.get('step')] = step.get('credit_level', 'other')
            success_credit_map[traj_uid] = step_credits
        
        advantages = batch.batch['advantages']
        response_mask = batch.batch['response_mask']
        traj_uids = batch.non_tensor_batch.get('traj_uid', None)
        step_indices = batch.non_tensor_batch.get('step_idx', None)
        episode_rewards = batch.non_tensor_batch.get('episode_rewards', None)
        
        if traj_uids is None or step_indices is None:
            print("[TrajEvolve] Warning: traj_uid or step_idx not found in batch, skipping step credit adjustment")
            return batch
        
        bsz = advantages.shape[0]
        adjusted_count = {'necessary_bonus': 0, 'error_penalty': 0, 'prefix_bonus': 0}
        
        for i in range(bsz):
            traj_uid = traj_uids[i]
            step_idx = int(step_indices[i])
            is_success = episode_rewards[i] > 0.5 if episode_rewards is not None else False
            
            adjustment = 0.0
            
            if is_success:
                if traj_uid in success_credit_map:
                    credit_level = success_credit_map[traj_uid].get(step_idx, 'other')
                    if credit_level == 'necessary':
                        adjustment = step_credit_value
                        adjusted_count['necessary_bonus'] += 1
            else:
                if traj_uid in failed_labels:
                    first_error = failed_labels[traj_uid].get('first_error_step', -1)
                    if step_idx < first_error:
                        adjustment = step_credit_value
                        adjusted_count['prefix_bonus'] += 1
                    elif step_idx == first_error:
                        adjustment = -step_credit_value
                        adjusted_count['error_penalty'] += 1
            
            if adjustment != 0.0:
                advantages[i] = advantages[i] + adjustment * response_mask[i]
        
        batch.batch['advantages'] = advantages
        
        print(f"[TrajEvolve] Step credit adjustment applied: "
              f"necessary_bonus={adjusted_count['necessary_bonus']}, "
              f"prefix_bonus={adjusted_count['prefix_bonus']}, "
              f"error_penalty={adjusted_count['error_penalty']}")
        
        return batch
    
    def _update_rewards_cache(self, trajectories: List[Dict]):
        """Append trajectory rewards to rewards_cache keyed by group uid."""
        for traj in trajectories:
            group_key = traj.get('uid')
            reward = traj.get('episode_reward', 0.0)
            if group_key:
                if group_key not in self.rewards_cache:
                    self.rewards_cache[group_key] = []
                self.rewards_cache[group_key].append(reward)
    
    def _compute_evolved_advantage(
        self,
        evolved_trajectories: List[Dict],
    ) -> List[Dict]:
        """Compute advantage = reward - mean(historical_rewards) for each evolved trajectory."""
        all_rewards = []
        for rewards in self.rewards_cache.values():
            all_rewards.extend(rewards)
        global_mean = np.mean(all_rewards) if all_rewards else 0.5
        
        for traj in evolved_trajectories:
            group_key = traj.get('uid')
            reward = traj.get('episode_reward', 0.0)
            
            if group_key and group_key in self.rewards_cache:
                historical_rewards = self.rewards_cache[group_key]
                baseline = np.mean(historical_rewards) if historical_rewards else global_mean
            else:
                baseline = global_mean
            
            traj['advantage'] = reward - baseline
            traj['baseline'] = baseline
            
            print(f"[TrajEvolve] Evolved trajectory {traj.get('traj_uid', 'unknown')}: "
                  f"reward={reward:.2f}, baseline={baseline:.2f}, advantage={traj['advantage']:.2f}")
        
        return evolved_trajectories
    
    def _train_on_evolved_trajectories(
        self,
        evolved_trajectories: List[Dict],
    ):
        """[DEPRECATED] Use _train_on_evolution_batch instead."""
        print("[TrajEvolve] _train_on_evolved_trajectories is deprecated, use _train_on_evolution_batch")
        return
    
    def _train_on_evolution_batch(
        self,
        total_batch_list: List[List[Dict]],
        total_infos: List[List[Dict]],
        traj_metadata: Dict[int, Dict],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        tool_callings: np.ndarray,
        success: Dict[str, np.ndarray],
        evolved_trajectories: List[Dict] = None,
    ):
        """Train actor on successful evolved trajectories.

        Merges step batches -> computes log_probs, reward, advantage -> updates actor.
        """
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.reward import compute_reward
        
        success_rate = success.get('success_rate', np.zeros(len(total_batch_list)))
        successful_indices = [i for i in range(len(total_batch_list)) if success_rate[i] > 0.5]
        
        if not successful_indices:
            print("[TrajEvolve] No successful evolved trajectories to train on")
            return
        
        print(f"[TrajEvolve] Training on {len(successful_indices)} successful evolved trajectories")
        
        # Collect active step batches from successful trajectories
        all_step_batches = []
        step_to_traj_idx = []
        
        for traj_idx in successful_indices:
            for step_idx, step_batch in enumerate(total_batch_list[traj_idx]):
                if step_batch.get('active_masks', False):
                    step_batch['step_idx'] = step_idx
                    step_batch['episode_rewards'] = episode_rewards[traj_idx]
                    step_batch['episode_lengths'] = episode_lengths[traj_idx]
                    step_batch['tool_callings'] = tool_callings[traj_idx]
                    
                    for key, value in success.items():
                        if 'success_rate' in key:
                            step_batch[key] = value[traj_idx]
                    
                    if 'uid' not in step_batch:
                        step_batch['uid'] = traj_metadata.get(traj_idx, {}).get('uid', f'unknown_{traj_idx}')
                    if 'traj_uid' not in step_batch:
                        step_batch['traj_uid'] = traj_metadata.get(traj_idx, {}).get('traj_uid', f'traj_{traj_idx}')
                    
                    all_step_batches.append(step_batch)
                    step_to_traj_idx.append(traj_idx)
        
        if not all_step_batches:
            print("[TrajEvolve] No step batches to train on")
            return
        
        print(f"[TrajEvolve] Total {len(all_step_batches)} steps from successful trajectories")
        
        try:
            merged_batch_dict = collate_fn(all_step_batches)
            batch = DataProto.from_single_dict(
                data=merged_batch_dict,
                meta_info={
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                }
            )
        except Exception as e:
            print(f"[TrajEvolve] Error merging batches: {e}")
            return
        
        batch.non_tensor_batch['_step_to_traj_idx'] = np.array(step_to_traj_idx)
        
        # Ensure is_action_valid exists (required by ray_trainer)
        if 'is_action_valid' not in batch.non_tensor_batch:
            batch.non_tensor_batch['is_action_valid'] = np.ones(len(all_step_batches), dtype=bool)
        
        from agent_system.multi_turn_rollout import adjust_batch
        original_size = len(batch)
        batch = adjust_batch(self.config, batch, mode="copy")
        adjusted_size = len(batch)
        if adjusted_size != original_size:
            print(f"[TrajEvolve] Adjusted batch size from {original_size} to {adjusted_size} for chunk alignment")
        step_to_traj_idx = batch.non_tensor_batch.pop('_step_to_traj_idx').tolist()
        
        batch.batch["response_mask"] = compute_response_mask(batch)
        
        if "think_token_mask" in batch.batch:
            response_length = batch.batch["responses"].size(1)
            think_mask = batch.batch["think_token_mask"][:, :response_length]
            batch.batch["response_mask"] = batch.batch["response_mask"] * (1 - think_mask)
            loss_mask = batch.batch["attention_mask"].clone()
            loss_mask[:, -response_length:] = loss_mask[:, -response_length:] * (1 - think_mask)
            batch.batch["loss_mask"] = loss_mask
            print(f"[TrajEvolve] Mask-think applied: masked {think_mask.sum().item()} think tokens across batch")
        
        # Compute old_log_probs
        print("[TrajEvolve] Computing old_log_probs...")
        try:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            batch = batch.union(old_log_prob)
            if "entropys" in batch.batch:
                batch.batch.pop("entropys")
        except Exception as e:
            print(f"[TrajEvolve] Error computing log_probs: {e}")
            return
        
        # Compute reward
        print("[TrajEvolve] Computing reward...")
        try:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
            batch.batch["token_level_scores"] = reward_tensor
            if reward_extra_infos_dict:
                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
        except Exception as e:
            print(f"[TrajEvolve] Error computing reward: {e}")
            return
        
        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
            batch, invalid_metrics = apply_invalid_action_penalty(
                batch,
                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
            )
            print(f"[TrajEvolve] Invalid action penalty applied: {invalid_metrics}")
        
        if self.config.algorithm.use_kl_in_reward:
            batch, _ = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
        
        # Compute evolution advantage
        print("[TrajEvolve] Computing evolution advantage...")
        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
        batch = self._compute_evolution_batch_advantage(
            batch=batch,
            step_to_traj_idx=step_to_traj_idx,
            traj_metadata=traj_metadata,
            episode_rewards=episode_rewards,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )

        # Compute ref_log_prob for KL loss
        if self.use_reference_policy:
            print("[TrajEvolve] Computing ref_log_prob for KL loss...")
            try:
                if not self.ref_in_actor:
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                else:
                    ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                batch = batch.union(ref_log_prob)
            except Exception as e:
                print(f"[TrajEvolve] Error computing ref_log_prob: {e}")
        
        # Update actor
        print("[TrajEvolve] Updating actor...")
        try:
            if "loss_mask" in batch.batch:
                batch.meta_info["multi_turn"] = True
            else:
                batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            actor_output = self.actor_rollout_wg.update_actor(batch)
            
            if hasattr(actor_output, 'meta_info') and 'metrics' in actor_output.meta_info:
                from verl.utils.metric import reduce_metrics
                actor_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                print(f"[TrajEvolve] Actor update metrics: {actor_metrics}")
        except Exception as e:
            print(f"[TrajEvolve] Error updating actor: {e}")
            import traceback
            traceback.print_exc()
            return

        print("[TrajEvolve] Evolution training completed successfully")
    
    def _compute_evolution_batch_advantage(
        self,
        batch: DataProto,
        step_to_traj_idx: List[int],
        traj_metadata: Dict[int, Dict],
        episode_rewards: np.ndarray,
        norm_adv_by_std_in_grpo: bool = True,
        epsilon: float = 1e-6,
    ) -> DataProto:
        """Compute advantage for evolution batch using historical + current rewards as baseline (GRPO-style)."""
        from collections import defaultdict
        
        batch_size = len(step_to_traj_idx)
        token_level_rewards = batch.batch["token_level_rewards"]
        response_mask = batch.batch["response_mask"]
        
        # Collect current batch rewards per group (one reward per trajectory, not per step)
        current_batch_rewards = defaultdict(list)
        seen_traj_idx = set()
        for i in range(batch_size):
            traj_idx = step_to_traj_idx[i]
            if traj_idx in seen_traj_idx:
                continue
            seen_traj_idx.add(traj_idx)
            
            group_key = traj_metadata.get(traj_idx, {}).get('uid')
            reward = episode_rewards[traj_idx]
            if group_key:
                current_batch_rewards[group_key].append(reward)

        # Compute combined mean/std per group (historical + current batch)
        group_stats = {}
        for group_key in current_batch_rewards:
            historical = list(self.rewards_cache.get(group_key, []))
            combined = historical + current_batch_rewards[group_key]
            
            if not combined:
                raise ValueError(f"group_key '{group_key}' has no rewards in cache or current batch")
            
            mean = np.mean(combined)
            std = np.std(combined) if len(combined) > 1 else 1.0
            group_stats[group_key] = (mean, std)
        
        # Compute per-step advantage
        advantages = torch.zeros_like(token_level_rewards)
        
        for i in range(batch_size):
            traj_idx = step_to_traj_idx[i]
            group_key = traj_metadata.get(traj_idx, {}).get('uid')
            reward = episode_rewards[traj_idx]
            
            if not group_key or group_key not in group_stats:
                raise ValueError(f"group_key '{group_key}' not found in computed stats")
            
            baseline, std = group_stats[group_key]
            
            if norm_adv_by_std_in_grpo:
                advantage = (reward - baseline) / (std + epsilon)
            else:
                advantage = reward - baseline
            
            advantages[i] = advantage * response_mask[i]
        
        batch.batch["advantages"] = advantages
        batch.batch["returns"] = advantages
        
        return batch
    
    def _prepare_evolution_actions(
        self,
        trajectories: List[Dict],
        analysis_results: List[Dict],
        failed_labels: Dict[str, Dict],
    ) -> Tuple[Dict[int, Dict[int, str]], Dict[int, Dict]]:
        """Prepare per-step evolution actions for each trajectory.

        Success trajectories: skip redundant steps, replay necessary ones.
        Failed trajectories: replay prefix before error; map format errors to admissible actions.

        Returns:
            (evolution_actions, traj_metadata): evolution_actions maps {traj_idx: {step_idx: action}}.
        """
        evolution_actions = {}
        traj_metadata = {}
        
        analysis_by_uid = {}
        for analysis in analysis_results:
            traj_uid = analysis.get('trajectory_info', {}).get('traj_uid')
            if traj_uid:
                analysis_by_uid[traj_uid] = analysis
        
        for traj_idx, traj in enumerate(trajectories):
            traj_uid = traj.get('traj_uid')
            is_success = traj.get('is_success', False)
            trajectory_steps = traj.get('trajectory', [])
            
            traj_metadata[traj_idx] = {
                'traj_uid': traj_uid,
                'uid': traj.get('uid'),
                'gamefile': traj.get('gamefile'),
                'task': traj.get('task', ''),
                'is_success': is_success,
                'selection_type': traj.get('selection_type', 'unknown'),
                'session_id': traj.get('session_id'),
                'env_seed': traj.get('env_seed'),
            }
            
            if is_success:
                analysis = analysis_by_uid.get(traj_uid, {})
                step_analysis = analysis.get('step_analysis', [])
                
                redundant_indices = set()
                for s in step_analysis:
                    if s.get('credit_level') == 'redundant':
                        redundant_indices.add(s.get('step'))
                
                evolution_actions[traj_idx] = {}
                evolved_step_idx = 0
                for orig_step_idx, step in enumerate(trajectory_steps):
                    if orig_step_idx in redundant_indices:
                        continue
                    if not step.get('is_action_valid', True):
                        continue
                    
                    action = step.get('action_text', '')
                    evolution_actions[traj_idx][evolved_step_idx] = action
                    evolved_step_idx += 1
            else:
                error_info = failed_labels.get(traj_uid, {})
                first_error = error_info.get('first_error_step', len(trajectory_steps) - 1)
                error_type = error_info.get('error_type', 'action_error')
                
                evolution_actions[traj_idx] = {}
                
                # Replay steps before the first error
                for step_idx in range(min(first_error, len(trajectory_steps))):
                    action = trajectory_steps[step_idx].get('action_text', '')
                    evolution_actions[traj_idx][step_idx] = action
                
                # For format errors, map the error step to an admissible action;
                # for logic errors, let the model generate freely
                if error_type == 'format_error' and first_error < len(trajectory_steps):
                    step = trajectory_steps[first_error]
                    action = step.get('action_text', '')
                    admissible = step.get('admissible_commands', [])
                    
                    if admissible:
                        if self.env_type == 'webshop':
                            mapped_action = map_to_admissible_webshop(action, admissible)
                        else:
                            from agent_system.environments.env_package.alfworld import map_to_admissible_alfworld
                            mapped_action = map_to_admissible_alfworld(action, admissible)
                    else:
                        mapped_action = action
                    
                    evolution_actions[traj_idx][first_error] = mapped_action
        
        return evolution_actions, traj_metadata
    
    def _build_evolution_gen_batch(
        self,
        trajectories: List[Dict],
    ) -> DataProto:
        """Build a DataProto gen_batch for evolution. Actual observations come from env.reset()."""
        from verl.utils.dataset.rl_dataset import collate_fn
        from tensordict import TensorDict
        
        batch_data = []
        for traj in trajectories:
            gamefile = traj.get('gamefile', '')
            task = traj.get('task', '')
            
            prompt_text = f"Task: {task}" if task else "Complete the task."
            
            encoded = self.tokenizer(
                prompt_text,
                padding='max_length',
                truncation=True,
                max_length=self.config.data.max_prompt_length,
                return_tensors='pt',
            )
            
            env_kwargs = {'gamefile': gamefile} if gamefile else {}
            
            row = {
                'input_ids': encoded['input_ids'][0],
                'attention_mask': encoded['attention_mask'][0],
                'raw_prompt': [{'role': 'user', 'content': prompt_text}],
                'data_source': self.env_type,
                'env_kwargs': env_kwargs,
            }
            batch_data.append(row)
        
        batch = collate_fn(batch_data)
        gen_batch = DataProto.from_single_dict(data=batch)
        gen_batch.meta_info = {
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        
        return gen_batch
    
    def _run_evolution_phase(
        self,
        epoch: int,
        logger=None,
        global_steps=None,
    ):
        """Run trajectory evolution: prepare actions, run evolution loop, train on results."""
        import uuid
        
        print(f"\n[TrajEvolve] Starting evolution phase at epoch {epoch}")
        
        accumulated_trajectories = getattr(self, 'accumulated_trajectories', [])
        if not accumulated_trajectories:
            print("[TrajEvolve] No accumulated trajectories for evolution")
            return
        
        print(f"[TrajEvolve] Using {len(accumulated_trajectories)} accumulated trajectories from last {self.evolve_freq} epochs")
        
        analysis_results = getattr(self, 'accumulated_analysis_results', [])
        failed_labels = getattr(self, 'accumulated_failed_labels', {})
        
        print(f"[TrajEvolve] Preparing evolution actions...")
        evolution_actions, traj_metadata = self._prepare_evolution_actions(
            trajectories=accumulated_trajectories,
            analysis_results=analysis_results,
            failed_labels=failed_labels,
        )
        
        gen_batch = self._build_evolution_gen_batch(accumulated_trajectories)
        
        total_evolution = sum(len(steps) for steps in evolution_actions.values())
        print(f"[TrajEvolve] Prepared {len(accumulated_trajectories)} trajectories, {total_evolution} evolution action steps")
        
        print(f"[TrajEvolve] Running evolution loop...")
        try:
            (total_batch_list, episode_rewards, episode_lengths, 
             success, traj_uid_array, tool_callings, total_infos) = self.traj_collector.evolution_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=self.actor_rollout_wg,
                evolution_actions=evolution_actions,
                traj_metadata=traj_metadata,
            )
            
            evolved_trajectories = []
            for i in range(len(total_batch_list)):
                is_success = success.get('success_rate', np.zeros(len(total_batch_list)))[i] > 0.5
                
                trajectory_steps = []
                for step_idx, info in enumerate(total_infos[i]):
                    trajectory_steps.append({
                        'step_number': step_idx,
                        'action_text': info.get('action_text', ''),
                        'observation_text': info.get('observation_text', ''),
                        'is_action_valid': info.get('is_action_valid', True),
                        'is_evolution_action': info.get('is_evolution_action', False),
                    })
                    if info.get('won', False):
                        break
                
                metadata = traj_metadata.get(i, {})
                evolved_traj = {
                    'traj_uid': traj_uid_array[i],
                    'original_traj_uid': metadata.get('traj_uid'),
                    'gamefile': metadata.get('gamefile'),
                    'task': metadata.get('task'),
                    'trajectory': trajectory_steps,
                    'is_success': is_success,
                    'episode_reward': float(episode_rewards[i]),
                    'episode_length': int(episode_lengths[i]),
                    'evolution_type': 'success_skip' if metadata.get('is_success') else 'failed_rollback',
                }
                evolved_trajectories.append(evolved_traj)
                
                print(f"[TrajEvolve] Evolved trajectory {i}: "
                      f"original_uid={metadata.get('traj_uid', 'unknown')[:8]}..., "
                      f"success={is_success}, steps={len(trajectory_steps)}")
            
            total_evolved = len(evolved_trajectories)
            successful_evolved = sum(1 for t in evolved_trajectories if t.get('is_success', False))
            failed_evolved = total_evolved - successful_evolved
            
            print(f"[TrajEvolve] Evolution phase complete: "
                  f"{total_evolved}/{len(accumulated_trajectories)} trajectories evolved, "
                  f"{successful_evolved} successful, {failed_evolved} failed")
            
            if total_batch_list:
                self._train_on_evolution_batch(
                    total_batch_list=total_batch_list,
                    total_infos=total_infos,
                    traj_metadata=traj_metadata,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    tool_callings=tool_callings,
                    success=success,
                    evolved_trajectories=evolved_trajectories,
                )
                
        except Exception as e:
            print(f"[TrajEvolve] Error in evolution phase: {e}")
            import traceback
            traceback.print_exc()
        
        # Clear accumulated storage for next cycle
        self.accumulated_trajectories = []
        self.accumulated_analysis_results = []
        self.accumulated_failed_labels = {}
        self.rewards_cache = {}
        print(f"[TrajEvolve] Cleared accumulated trajectories and rewards cache for next evolution cycle")
    
    def _prepare_trajectories(
        self,
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
        success: Dict[str, np.ndarray],
        total_infos: List[List[Dict]],
        traj_uid: np.ndarray = None,
        uid: np.ndarray = None,
        max_per_group: int = 2,
    ) -> tuple:
        """Extract trajectories from rollout data and sample candidates for evolution.

        Phase 1: extract trajectory dicts from raw rollout data.
        Phase 2: per-group sampling prioritising high invalid_ratio.

        Returns:
            (trajectories, evolution_candidates)
        """
        from collections import defaultdict
        
        trajectories = []
        batch_size = len(total_infos) if total_infos else 0
        
        if batch_size == 0:
            print("[TrajEvolve] Warning: No trajectories to extract")
            return trajectories, []
        
        per_sample_success = success.get('success_rate', np.zeros(batch_size))
        
        for i in range(batch_size):
            try:
                infos_list = total_infos[i] if i < len(total_infos) else []
                
                result = {
                    'traj_uid': traj_uid[i] if traj_uid is not None and i < len(traj_uid) else None,
                    'uid': uid[i] if uid is not None and i < len(uid) else None,
                    'episode_reward': float(episode_rewards[i]),
                    'episode_length': int(episode_lengths[i]),
                    'is_success': bool(per_sample_success[i] > 0.5),
                }
                
                if infos_list and len(infos_list) > 0:
                    result['task'] = infos_list[0].get('task', '')
                    result['gamefile'] = infos_list[0].get('extra.gamefile', None)
                    result['session_id'] = infos_list[0].get('session_id', None)
                    result['env_seed'] = infos_list[0].get('env_seed', None)
                else:
                    result['task'] = ''
                    result['gamefile'] = None
                    result['session_id'] = None
                    result['env_seed'] = None
                
                result['trajectory'] = []
                for step_idx, info in enumerate(infos_list):
                    step_result = {
                        'step_number': step_idx,
                        'action_text': info.get('action_text', ''),
                        'observation_text': info.get('observation_text', ''),
                        'is_action_valid': bool(info.get('is_action_valid', True)),
                        'admissible_commands': info.get('admissible_commands', []),
                    }
                    result['trajectory'].append(step_result)
                    if info.get('won', False):
                        break
                
                trajectories.append(result)
                
            except Exception as e:
                print(f"[TrajEvolve] Error extracting trajectory {i}: {e}")
                continue
        
        # Phase 2: per-group sampling (sort by invalid_ratio descending)
        def _sort_key(traj: Dict) -> tuple:
            steps = traj.get('trajectory', [])
            n = len(steps)
            inv = sum(1 for s in steps if not s.get('is_action_valid', True)) / n if n else 0.0
            return (inv, n)
        
        groups = defaultdict(list)
        for traj in trajectories:
            groups[traj.get('uid', 'unknown')].append(traj)
        
        evolution_candidates = []
        for group_key, group_trajs in groups.items():
            pos = sorted([t for t in group_trajs if t.get('is_success', False)],
                         key=_sort_key, reverse=True)
            neg = sorted([t for t in group_trajs if not t.get('is_success', False)],
                         key=_sort_key, reverse=True)
            
            picks = []
            for t in pos[:max_per_group]:
                c = t.copy(); c['selection_type'] = 'success'; picks.append(c)
            remaining = max_per_group - len(picks)
            for t in neg[:remaining]:
                c = t.copy(); c['selection_type'] = 'failed'; picks.append(c)
            
            evolution_candidates.extend(picks)
            
            n_success = sum(1 for p in picks if p['selection_type'] == 'success')
            n_failed = len(picks) - n_success
            if picks:
                print(f"[TrajEvolve] Group {group_key}: {n_success} success + {n_failed} failed")
            else:
                print(f"[TrajEvolve] Group {group_key}: no trajectories available")
        
        print(f"[TrajEvolve] Total selected: {len(evolution_candidates)} from {len(groups)} groups")
        return trajectories, evolution_candidates
    
    def fit(self):
        """Main training loop with trajectory evolution every evolve_freq epochs."""
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking
        from verl.trainer.ppo.core_algos import agg_loss
        from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
        from verl.utils.metric import reduce_metrics
        from verl.trainer.ppo.reward import compute_reward, compute_reward_async

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer("step", timing_raw):
                    with _timer("gen", timing_raw):
                        gen_batch_output, traj_info = self.traj_collector.multi_turn_loop(
                            gen_batch=gen_batch,
                            actor_rollout_wg=self.actor_rollout_wg,
                            envs=self.envs,
                            is_train=True,
                            with_traj_info=True,
                        )

                    del batch
                    batch = gen_batch_output
                    
                    # TrajEvolve analysis: extract trajectories, analyze credit, train discriminator, label failures
                    with _timer("traj_evolve_analysis", timing_raw):
                        total_batch_list = traj_info.get('total_batch_list', [])
                        uid_array = np.array([
                            total_batch_list[i][0]['uid'] if total_batch_list[i] else None
                            for i in range(len(total_batch_list))
                        ], dtype=object) if total_batch_list else None
                        
                        trajectories, evolution_candidates = self._prepare_trajectories(
                            episode_rewards=traj_info['episode_rewards'],
                            episode_lengths=traj_info['episode_lengths'],
                            success=traj_info['success'],
                            total_infos=traj_info['total_infos'],
                            traj_uid=traj_info['traj_uid'],
                            uid=uid_array,
                        )
                        
                        analysis_results, success_labels = self._analyze_successful_trajectories(trajectories)
                        
                        selected_failed = [t for t in evolution_candidates if not t.get('is_success', False)]

                        failed_labels = {}
                        if selected_failed:
                            self._train_discriminator(analysis_results, epoch)
                            failed_labels = self._label_failed_trajectories(selected_failed)
                        else:
                            print(f"[TrajEvolve] No failed trajectories selected, skip discriminator training")
                        
                        # Accumulate for evolution phase
                        self.accumulated_trajectories.extend(evolution_candidates)
                        self.accumulated_analysis_results.extend(analysis_results)
                        self.accumulated_failed_labels.update(failed_labels)
                        self._update_rewards_cache(trajectories)

                    # GiGPO step rewards
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor

                    from agent_system.multi_turn_rollout import adjust_batch
                    batch = adjust_batch(self.config, batch)
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with _timer("reward", timing_raw):
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_result = self.traj_evolve_reward_fn(
                                batch,
                                success_labels=success_labels,
                            )
                            reward_tensor = reward_result["reward_tensor"]
                            reward_extra_infos_dict = reward_result.get("reward_extra_info", {})

                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                            batch, invalid_metrics = apply_invalid_action_penalty(
                                batch,
                                invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                            )
                            metrics.update(invalid_metrics)

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                        )
                        
                        if self.step_credit_value > 0:
                            batch = self._apply_step_credit_adjustment(
                                batch,
                                analysis_results=analysis_results,
                                failed_labels=failed_labels,
                                step_credit_value=self.step_credit_value,
                            )

                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with _timer("update_actor", timing_raw):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    if (epoch + 1) % self.evolve_freq == 0:
                        with _timer("evolution_phase", timing_raw):
                            self._run_evolution_phase(epoch=epoch, logger=logger, global_steps=self.global_steps)

                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                metrics.update({
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                })
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    self._save_validation_results_to_csv(last_val_metrics)
                    progress_bar.close()
                    return

    def _save_validation_results_to_csv(self, val_metrics: dict):
        """Save final validation results to CSV files, organized by environment type."""
        import csv
        from datetime import datetime
        
        result_dir = os.path.join(os.path.expanduser("~"), "result")
        experiment_name = self.config.trainer.get("experiment_name", "default")
        
        # Extract model info from experiment_name (e.g. "grpo_qwen2.5_1.5b" -> "qwen2.5_1.5b")
        parts = experiment_name.split('_', 1)
        model_info = parts[1] if len(parts) > 1 else experiment_name
        result_dir = os.path.join(result_dir, model_info)
        os.makedirs(result_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        if "alfworld" in self.config.env.env_name.lower():
            dataset_name = "alfworld"
            if self.config.env.alfworld.get('validate_both_datasets', False):
                val_sets = ["eval_in_distribution", "eval_out_of_distribution"]
            else:
                val_sets = [self.config.env.alfworld.eval_dataset]
            task_order = [
                "pick_and_place",
                "look_at_obj_in_light",
                "pick_clean_then_place_in_recep",
                "pick_heat_then_place_in_recep",
                "pick_cool_then_place_in_recep",
                "pick_two_obj_and_place",
            ]
        elif "webshop" in self.config.env.env_name.lower():
            dataset_name = "webshop"
            val_sets = [None]
            task_order = []
        elif "sokoban" in self.config.env.env_name.lower():
            dataset_name = "sokoban"
            val_sets = [None]
            task_order = []
        else:
            dataset_name = "unknown"
            val_sets = [None]
            task_order = []
        
        def extract_metrics(metric_type, suffix=""):
            """Extract metrics of a specific type for all datasets and tasks."""
            data = {}
            metric_key = metric_type if not suffix else f"{metric_type}{suffix}"
            
            for val_set in val_sets:
                if val_set is not None:
                    key = f"val/{val_set}/{metric_key}"
                else:
                    key = f"val/{metric_key}"
                if key in val_metrics:
                    if metric_key not in data:
                        data[metric_key] = {}
                    value = val_metrics[key]
                    if metric_type == "success_rate":
                        value *= 100
                    data[metric_key][val_set if val_set else "default"] = value
            
            if task_order:
                for task in task_order:
                    task_key = f"{task}_{metric_key}"
                    for val_set in val_sets:
                        if val_set is not None:
                            key = f"val/{val_set}/{task_key}"
                        else:
                            key = f"val/{task_key}"
                        if key in val_metrics:
                            if task_key not in data:
                                data[task_key] = {}
                            value = val_metrics[key]
                            if metric_type == "success_rate":
                                value *= 100
                            data[task_key][val_set if val_set else "default"] = value
            
            return data
        
        success_rate_data = extract_metrics("success_rate")
        episode_length_data = extract_metrics("episode", "/length")
        success_episode_length_data = extract_metrics("success_episode", "/length")
        
        print(f"DEBUG: success_rate_data keys = {success_rate_data.keys()}")
        print(f"DEBUG: val_metrics keys with 'success_rate' = {[k for k in val_metrics.keys() if 'success_rate' in k]}")
        
        if "webshop" in self.config.env.env_name.lower():
            task_score_data = extract_metrics("webshop_task_score (not success_rate)")
            
            all_metrics = {}
            if success_rate_data and "success_rate" in success_rate_data:
                all_metrics["success_rate"] = success_rate_data["success_rate"].get("default", "")
            if success_episode_length_data and "success_episode/length" in success_episode_length_data:
                all_metrics["success_episode/length"] = success_episode_length_data["success_episode/length"].get("default", "")
            if episode_length_data and "episode/length" in episode_length_data:
                all_metrics["episode/length"] = episode_length_data["episode/length"].get("default", "")
            if task_score_data and "webshop_task_score (not success_rate)" in task_score_data:
                all_metrics["webshop_task_score"] = task_score_data["webshop_task_score (not success_rate)"].get("default", "")
            
            csv_filename = f"{dataset_name}_metrics_{experiment_name}_{timestamp}.csv"
            csv_path = os.path.join(result_dir, csv_filename)
            self._write_webshop_csv(csv_path, all_metrics)
            print(f"Saved webshop metrics to {csv_path}")
        else:
            if success_rate_data:
                csv_filename = f"{dataset_name}_success_rate_{experiment_name}_{timestamp}.csv"
                csv_path = os.path.join(result_dir, csv_filename)
                self._write_csv(csv_path, success_rate_data, task_order, val_sets, "success_rate")
                print(f"Saved success_rate results to {csv_path}")
            
            if episode_length_data:
                csv_filename = f"{dataset_name}_episode_length_{experiment_name}_{timestamp}.csv"
                csv_path = os.path.join(result_dir, csv_filename)
                self._write_csv(csv_path, episode_length_data, task_order, val_sets, "episode/length")
                print(f"Saved episode/length results to {csv_path}")
            
            if success_episode_length_data:
                csv_filename = f"{dataset_name}_success_episode_length_{experiment_name}_{timestamp}.csv"
                csv_path = os.path.join(result_dir, csv_filename)
                self._write_csv(csv_path, success_episode_length_data, task_order, val_sets, "success_episode/length")
                print(f"Saved success_episode/length results to {csv_path}")

    def _write_csv(self, csv_path, data, task_order, val_sets, metric_type):
        """Write metrics data to CSV with rows: [val_set_names], [metric_names], [values]."""
        import csv
        
        actual_val_sets = [v if v is not None else "default" for v in val_sets]
        
        task_columns = []
        for task in task_order:
            if metric_type == "success_rate":
                task_columns.append(f"{task}_success_rate")
            elif metric_type == "episode/length":
                task_columns.append(f"{task}_episode/length")
            elif metric_type == "success_episode/length":
                task_columns.append(f"{task}_success_episode/length")

        if metric_type == "success_rate":
            task_columns.append("success_rate")
        elif metric_type == "episode/length":
            task_columns.append("episode/length")
        elif metric_type == "success_episode/length":
            task_columns.append("success_episode/length")
        elif metric_type == "task_score":
            task_columns.append("webshop_task_score (not success_rate)")
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            dataset_row = [""]
            for val_set_name in actual_val_sets:
                for _ in task_columns:
                    dataset_row.append(val_set_name)
            writer.writerow(dataset_row)
            
            metric_row = [""]
            for val_set_name in actual_val_sets:
                for col in task_columns:
                    metric_row.append(col)
            writer.writerow(metric_row)
            
            data_row = [""]
            for val_set_name in actual_val_sets:
                for col in task_columns:
                    if col in data and val_set_name in data[col]:
                        value = data[col][val_set_name]
                        data_row.append(str(value))
                    else:
                        data_row.append("")
            writer.writerow(data_row)

    def _write_webshop_csv(self, csv_path, all_metrics):
        """Write WebShop metrics to a single CSV with one header row and one data row."""
        import csv
        
        column_order = ["success_rate", "success_episode/length", "episode/length", "webshop_task_score"]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            header_row = [""] + column_order
            writer.writerow(header_row)
            
            data_row = [""]
            for col in column_order:
                value = all_metrics.get(col, "")
                data_row.append(str(value) if value != "" else "")
            writer.writerow(data_row)
