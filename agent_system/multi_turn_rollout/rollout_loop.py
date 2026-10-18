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

import copy
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import process_image, to_list_of_dict, torch_to_numpy, filter_group_data
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict, Optional, Tuple
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
import re
import ray
import random



def extract_clean_action(action_text: str) -> str:
    """Extract the action string from <action>...</action> tags, or return raw text."""
    match = re.search(r'<action>(.*?)</action>', action_text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return action_text


def build_think_only_obs(obs_text: str, evolution_action: str, env_type: str = 'alfworld') -> str:
    """Modify observation for think-only generation by removing action instructions and adding a constraint.

    The constraint specifies the predetermined action and asks the model to generate reasoning only.
    """
    if env_type == 'webshop':
        cut_marker = "Now it's your turn to take one action for the current step."
    else:
        cut_marker = "Now it's your turn to take an action."
    
    if cut_marker in obs_text:
        obs_text = obs_text[:obs_text.find(cut_marker)]
    
    if env_type == 'webshop':
        constraint = (
            f"The action you decide to take is: {evolution_action}\n"
            f"Generate only your reasoning content about this action."
            f"Do not output any action tags, just the reasoning content."
        )
    else:
        constraint = (
            f"The action to take is: {evolution_action}\n"
            f"Generate only your reasoning for why this action is appropriate. "
            f"Do not output any action tags, just the reasoning content."
        )
    
    return obs_text.strip() + "\n\n" + constraint


def extract_think_content(response: str) -> str:
    """Extract content from <think>...</think> tags, or return cleaned response without action tags."""
    think_match = re.search(r'<think>\s*(.*?)\s*</think>', response, flags=re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    response = re.sub(r'<action>.*?</action>', '', response, flags=re.DOTALL)
    response = re.sub(r'<think>|</think>', '', response)
    
    return response.strip()


@ray.remote
def _create_env_and_reset(gamefile: str) -> Tuple[str, Dict]:
    """Create a single ALFWorld environment from gamefile, reset it, and return (obs_text, info)."""
    from agent_system.environments.env_package.alfworld import create_single_game_env
    
    if not gamefile:
        return '', {}
    try:
        env = create_single_game_env(gamefile)
        obs, info = env.reset()
        obs_text = obs if isinstance(obs, str) else str(obs)
        env.close()
        return obs_text, info
    except Exception as e:
        print(f"[Evolution] Failed to create env for {gamefile}: {e}")
        return '', {}


@ray.remote
def _step_single_env(gamefile: str, action: str, env_state_actions: List[str]) -> Tuple[str, float, bool, Dict]:
    """Recreate env, replay previous actions to restore state, then execute current action."""
    from agent_system.environments.env_package.alfworld import create_single_game_env
    
    if not gamefile:
        return '', 0.0, True, {'is_action_valid': False}
    
    try:
        env = create_single_game_env(gamefile)
        obs, _ = env.reset()
        
        for prev_action in env_state_actions:
            obs, _, done, info = env.step(prev_action)
            if done:
                break
        
        obs, reward, done, info = env.step(action)
        obs_text = obs if isinstance(obs, str) else str(obs)
        
        admissible = []
        if hasattr(env, 'get_admissible_commands'):
            admissible = list(env.get_admissible_commands())
        info['admissible_commands'] = admissible
        
        env.close()
        return obs_text, float(reward), bool(done), info
        
    except Exception as e:
        print(f"[Evolution] Error stepping env: {e}")
        return '', 0.0, True, {'is_action_valid': False}


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """Initialize with config, tokenizer, and optional image processor."""
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """Process a single observation into model-ready format (input_ids, attention_mask, etc.)."""

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        chat = np.array([{
            "content": obs_content,
            "role": "user",
        }])
        
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs
        )
        
        row_dict = {}
        
        if is_multi_modal:
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """Process a batch of observations into a DataProto with model-ready tensors."""
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        for item in range(batch_size):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        batch = collate_fn(processed_samples)
        
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            total_infos: List[List[Dict]] = None,
            ) -> DataProto:
        """Collect active steps from all trajectories into a single DataProto for training."""
        batch_size = len(total_batch_list)

        success_rate = {}
        for key, value in success.items():
            if 'success_rate' in key:
                success_rate[key] = np.mean(value)
        
        episode_length_array = success.get('episode_length', None)
        success_episode_length_array = success.get('success_episode_length', None)
        
        task_episode_lengths = {}
        task_success_episode_lengths = {}
        for key, value in success.items():
            if '_episode_length' in key and 'success_episode_length' not in key and key != 'episode_length':
                task_episode_lengths[key] = np.array(value)
            elif '_success_episode_length' in key and key != 'success_episode_length':
                task_success_episode_lengths[key] = np.array(value)
        
        effective_batch = []
        for bs in range(batch_size):
            for step_idx, data in enumerate(total_batch_list[bs]):
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    data['step_idx'] = step_idx
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_lengths'] = episode_lengths[bs]
                    data['tool_callings'] = tool_callings[bs]
                    for key, value in success_rate.items():
                        data[key] = value
                    if episode_length_array is not None:
                        data['_episode_length_array'] = episode_length_array
                    if success_episode_length_array is not None:
                        data['_success_episode_length_array'] = success_episode_length_array
                    for key, value in task_episode_lengths.items():
                        data[f'_{key}_array'] = value
                    for key, value in task_success_episode_lengths.items():
                        data[f'_{key}_array'] = value

                    effective_batch.append(data)
            
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        
        # total_infos is trajectory-level (not step-level), returned via traj_info in multi_turn_loop
        
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ):
        """Collect trajectories through parallel agent-environment interaction.

        Returns:
            (total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_infos)
        """

        batch_size = len(gen_batch.batch)

        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        
        # Extract task description from initial observation
        tasks = []
        initial_obs_texts = obs.get('text', [])
        for i in range(batch_size):
            obs_text = initial_obs_texts[i] if i < len(initial_obs_texts) else ''
            task_start = obs_text.find('Your task is to: ')
            if task_start != -1:
                task_text = obs_text[task_start + len('Your task is to: '):]
                end_idx = task_text.find('\n') if '\n' in task_text else len(task_text)
                tasks.append(task_text[:end_idx].strip())
            else:
                raise ValueError(f"Cannot find 'Your task is to: ' in observation text: {obs_text}")

        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            step_input_ids = batch.batch['input_ids'].clone()

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            prompt_texts = self.tokenizer.batch_decode(step_input_ids, skip_special_tokens=True)
            
            # Get admissible commands before step (ALFWorld or WebShop)
            current_admissible_commands = None
            current_available_actions = None
            
            if hasattr(envs.envs, 'get_admissible_commands'):
                admissible_commands_list = envs.envs.get_admissible_commands
                current_admissible_commands = [list(cmds) if cmds else [] for cmds in admissible_commands_list]
            elif hasattr(envs.envs, 'get_available_actions'):
                available_actions_list = envs.envs.get_available_actions
                current_available_actions = copy.deepcopy(available_actions_list)
            
            next_obs, rewards, dones, infos = envs.step(text_actions)
            
            for i, info in enumerate(infos):
                info['action_text'] = text_actions[i]
                info['task'] = tasks[i] if tasks else ''
                info['prompt_text'] = prompt_texts[i] if prompt_texts else ''
                if current_admissible_commands:
                    info['admissible_commands'] = current_admissible_commands[i]
                elif current_available_actions:
                    info['admissible_commands'] = current_available_actions[i]
                else:
                    raise ValueError(f"No admissible_commands or available_actions found for env {i}")

            
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]

            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_infos
    
    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """Repeatedly run vanilla_multi_turn_loop until the target batch size is met (DAPO-style dynamic sampling).

        Returns:
            (total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_infos)
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_infos = []
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            result = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, infos = result
            
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)
            
            # Sync-filter infos with the filtered batch_list via traj_uid
            if infos is not None:
                filtered_traj_uid_set = set(traj_uid.tolist())
                for i, original_uid in enumerate(result[4].tolist()):
                    if original_uid in filtered_traj_uid_set:
                        total_infos.append(infos[i])

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_infos

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            with_traj_info: bool = False,
            ):
        """Dispatch to dynamic or vanilla rollout loop, then gather into DataProto.

        Returns DataProto, or (DataProto, traj_info_dict) if with_traj_info=True.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
            
        if self.config.algorithm.filter_groups.enable and is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_infos = self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        else:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_infos = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
        
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)
        

        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
            total_infos=total_infos,
        )
        
        if with_traj_info:
            traj_info = {
                'total_batch_list': total_batch_list,
                'episode_rewards': total_episode_rewards,
                'episode_lengths': total_episode_lengths,
                'success': total_success,
                'traj_uid': total_traj_uid,
                'total_infos': total_infos,
            }
            return gen_batch_output, traj_info
        
        return gen_batch_output

    def evolution_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            evolution_actions: Dict[int, Dict[int, str]],
            traj_metadata: Dict[int, Dict],
            ):
        """Trajectory evolution loop with independent per-trajectory environments.

        For evolution-action steps: modifies prompt for think-only generation,
        uses predetermined action, and wraps output with proper tags.
        For free steps: normal model inference.

        Returns:
            (total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_infos)
        """
        batch_size = len(gen_batch.batch)
        
        env_type = 'webshop' if 'webshop' in self.config.env.env_name.lower() else 'alfworld'
        
        # Build evolution environments (env-specific)
        if env_type == 'webshop':
            from agent_system.environments.env_package.webshop import build_evolution_envs_webshop
            seed_goal_pairs = [
                (traj_metadata.get(i, {}).get('env_seed', 0),
                 int(traj_metadata.get(i, {}).get('session_id', 0)))
                for i in range(batch_size)
            ]
            evolution_envs = build_evolution_envs_webshop(seed_goal_pairs, self.config)
        else:
            from agent_system.environments.env_package.alfworld import build_evolution_envs_alfworld
            gamefiles = [traj_metadata.get(i, {}).get('gamefile', '') for i in range(batch_size)]
            evolution_envs = build_evolution_envs_alfworld(gamefiles, self.config)
        
        obs, infos = evolution_envs.reset(kwargs=None)
        
        if not isinstance(infos, list):
            infos = [infos]
        
        lenght_obs = len(obs['text']) if obs['text'] is not None else 0
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        # Use original uid for correct GRPO grouping
        uid_batch = np.array([traj_metadata[i]['uid'] for i in range(batch_size)], dtype=object)
        
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        
        tasks = []
        initial_obs_texts = obs.get('text', [])
        for i in range(batch_size):
            obs_text = initial_obs_texts[i] if i < len(initial_obs_texts) else ''
            task_start = obs_text.find('Your task is to: ')
            if task_start != -1:
                task_text = obs_text[task_start + len('Your task is to: '):]
                end_idx = task_text.find('\n') if '\n' in task_text else len(task_text)
                tasks.append(task_text[:end_idx].strip())
            else:
                raise ValueError(f"Cannot find 'Your task is to: ' in observation text: {obs_text}")
        
        current_steps = np.zeros(batch_size, dtype=int)
        
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            
            # Determine which trajectories have a predetermined evolution action at current step
            has_evolution_action = np.zeros(batch_size, dtype=bool)
            evolution_action_list = [None] * batch_size
            
            for i in range(batch_size):
                if i in evolution_actions and current_steps[i] in evolution_actions[i]:
                    has_evolution_action[i] = True
                    evolution_action_list[i] = evolution_actions[i][current_steps[i]]
            
            original_obs_texts = list(obs['text']) if obs.get('text') is not None else None
            
            # Modify observations for think-only generation where applicable
            modified_obs = obs.copy()
            if obs.get('text') is not None:
                modified_obs_texts = list(obs['text'])
                for i in range(batch_size):
                    if has_evolution_action[i] and active_masks[i]:
                        modified_obs_texts[i] = build_think_only_obs(
                            modified_obs_texts[i], 
                            evolution_action_list[i],
                            env_type=env_type
                        )
                modified_obs['text'] = modified_obs_texts
            
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=modified_obs)
            
            # Also preprocess original obs for rebuilding evolution steps later
            original_batch = None
            if original_obs_texts is not None and any(has_evolution_action):
                original_obs_for_batch = obs.copy()
                original_obs_for_batch['text'] = original_obs_texts
                original_batch = self.preprocess_batch(gen_batch=gen_batch, obs=original_obs_for_batch)
            
            step_input_ids = batch.batch['input_ids'].clone()

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input, actor_rollout_wg.world_size)
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            batch_output = unpad_dataproto(batch_output_padded, pad_size=pad_size)

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            # For evolution actions: extract think content and wrap with predetermined action
            final_actions = []
            for i in range(batch_size):
                if has_evolution_action[i] and active_masks[i]:
                    think_content = extract_think_content(text_actions[i])
                    action = evolution_action_list[i]
                    final_action = f"<think>\n{think_content}\n</think>\n<action>{action}</action>"
                    final_actions.append(final_action)
                else:
                    final_actions.append(text_actions[i])
            
            text_actions = final_actions
            
            prompt_texts = self.tokenizer.batch_decode(step_input_ids, skip_special_tokens=True)
            
            current_admissible_commands = None
            current_available_actions = None
            
            if hasattr(evolution_envs.envs, 'get_admissible_commands'):
                admissible_commands_list = evolution_envs.envs.get_admissible_commands
                current_admissible_commands = [list(cmds) if cmds else [] for cmds in admissible_commands_list]
            elif hasattr(evolution_envs.envs, 'get_available_actions'):
                available_actions_list = evolution_envs.envs.get_available_actions
                current_available_actions = copy.deepcopy(available_actions_list)
            
            next_obs, rewards, dones, infos = evolution_envs.step(list(text_actions))
            
            for i, info in enumerate(infos):
                info['action_text'] = extract_clean_action(text_actions[i])
                info['task'] = tasks[i] if tasks else ''
                info['prompt_text'] = prompt_texts[i] if prompt_texts else ''
                if current_admissible_commands:
                    info['admissible_commands'] = current_admissible_commands[i]
                elif current_available_actions:
                    info['admissible_commands'] = current_available_actions[i]
                else:
                    raise ValueError(f"No admissible_commands or available_actions found for env {i}")
                info['is_evolution_action'] = has_evolution_action[i]

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Rebuild batch data for evolution-action steps using original prompt + full response
            # verl expects input_ids = [left-padded prompt | right-padded response]
            response_max_len_for_mask = batch.batch['responses'].shape[1]
            think_token_mask = torch.zeros(batch_size, response_max_len_for_mask, dtype=torch.long)

            if original_batch is not None:
                for i in range(batch_size):
                    if has_evolution_action[i] and active_masks[i]:
                        original_prompt_ids = original_batch.batch['input_ids'][i]
                        original_attention_mask = original_batch.batch['attention_mask'][i]
                        
                        prompt_len = original_attention_mask.sum().item()
                        original_prompt_ids_trimmed = original_prompt_ids[-prompt_len:]
                        
                        pad_token_id = self.tokenizer.pad_token_id
                        eos_token_id = self.tokenizer.eos_token_id
                        
                        prompts_max_len = batch.batch['prompts'].shape[1] if 'prompts' in batch.batch else batch.batch['input_ids'].shape[1] - batch.batch['responses'].shape[1]
                        response_max_len = batch.batch['responses'].shape[1]
                        
                        # --- Budget-aware response encoding ---
                        # Encode parts separately so we can truncate think while preserving action + EOS
                        action = evolution_action_list[i]
                        think_content = extract_think_content(text_actions[i])
                        
                        think_prefix_ids = self.tokenizer.encode("<think>\n", add_special_tokens=False, return_tensors='pt')[0]
                        action_suffix_ids = self.tokenizer.encode(f"\n</think>\n<action>{action}</action>", add_special_tokens=False, return_tensors='pt')[0]
                        eos_ids = torch.tensor([eos_token_id], dtype=torch.long)
                        
                        # Calculate max think token budget (reserve space for prefix + suffix + EOS)
                        reserved_len = len(think_prefix_ids) + len(action_suffix_ids) + len(eos_ids)
                        max_think_tokens = response_max_len - reserved_len
                        
                        think_ids = self.tokenizer.encode(think_content, add_special_tokens=False, return_tensors='pt')[0] if think_content else torch.tensor([], dtype=torch.long)
                        if len(think_ids) > max_think_tokens:
                            think_ids = think_ids[:max(0, max_think_tokens)]
                        
                        # Assemble: <think>\n{think}\n</think>\n<action>{action}</action><EOS>
                        response_ids = torch.cat([think_prefix_ids, think_ids, action_suffix_ids, eos_ids])
                        
                        think_len = len(think_prefix_ids) + len(think_ids)
                        think_token_mask[i, :min(think_len, response_max_len_for_mask)] = 1
                        
                        # Build prompt portion (left-padded)
                        if len(original_prompt_ids_trimmed) < prompts_max_len:
                            prompt_pad_len = prompts_max_len - len(original_prompt_ids_trimmed)
                            prompt_portion = torch.cat([
                                torch.full((prompt_pad_len,), pad_token_id, dtype=torch.long),
                                original_prompt_ids_trimmed
                            ])
                            prompt_mask = torch.cat([
                                torch.zeros(prompt_pad_len, dtype=torch.long),
                                torch.ones(len(original_prompt_ids_trimmed), dtype=torch.long)
                            ])
                        elif len(original_prompt_ids_trimmed) > prompts_max_len:
                            prompt_portion = original_prompt_ids_trimmed[-prompts_max_len:]
                            prompt_mask = torch.ones(prompts_max_len, dtype=torch.long)
                        else:
                            prompt_portion = original_prompt_ids_trimmed
                            prompt_mask = torch.ones(prompts_max_len, dtype=torch.long)
                        
                        # Build response portion (right-padded)
                        if len(response_ids) < response_max_len:
                            response_pad_len = response_max_len - len(response_ids)
                            response_portion = torch.cat([
                                response_ids,
                                torch.full((response_pad_len,), pad_token_id, dtype=torch.long)
                            ])
                            response_mask_portion = torch.cat([
                                torch.ones(len(response_ids), dtype=torch.long),
                                torch.zeros(response_pad_len, dtype=torch.long)
                            ])
                        elif len(response_ids) > response_max_len:
                            response_portion = response_ids[:response_max_len]
                            response_mask_portion = torch.ones(response_max_len, dtype=torch.long)
                        else:
                            response_portion = response_ids
                            response_mask_portion = torch.ones(response_max_len, dtype=torch.long)
                        
                        # Combine: input_ids = [prompt_portion | response_portion]
                        new_input_ids = torch.cat([prompt_portion, response_portion])
                        new_attention_mask = torch.cat([prompt_mask, response_mask_portion])
                        
                        batch.batch['input_ids'][i] = new_input_ids
                        batch.batch['attention_mask'][i] = new_attention_mask
                        batch.batch['responses'][i] = response_portion
                        
                        if 'prompts' in batch.batch:
                            batch.batch['prompts'][i] = prompt_portion
                        
                        if 'position_ids' in batch.batch:
                            new_position_ids = torch.zeros_like(new_attention_mask, dtype=torch.long)
                            valid_positions = new_attention_mask.cumsum(dim=0) - 1
                            new_position_ids = torch.where(new_attention_mask > 0, valid_positions, torch.zeros_like(valid_positions))
                            batch.batch['position_ids'][i] = new_position_ids
                        
                        # Zero out rollout_log_probs (PPO will recompute old_log_probs)
                        if 'rollout_log_probs' in batch.batch:
                            logprob_max_len = batch.batch['rollout_log_probs'].shape[1]
                            batch.batch['rollout_log_probs'][i] = torch.zeros(logprob_max_len, dtype=batch.batch['rollout_log_probs'].dtype)
            
            batch.batch['think_token_mask'] = think_token_mask

            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])
                if active_masks[i]:
                    current_steps[i] += 1

            is_done = np.logical_or(is_done, dones)
            obs = next_obs

            if is_done.all():
                break
        
        # Compute success: find the last active step for each trajectory and check 'won'
        success_array = np.zeros(batch_size, dtype=np.float32)
        for i in range(batch_size):
            if total_infos[i] and total_batch_list[i]:
                for j in reversed(range(len(total_batch_list[i]))):
                    batch_item = total_batch_list[i][j]
                    if batch_item.get('active_masks', False):
                        info = total_infos[i][j]
                        won = info.get('won', False)
                        if isinstance(won, list):
                            won = won[0] if won else False
                        success_array[i] = 1.0 if won else 0.0
                        break
        success = {'success_rate': success_array}
        
        evolution_envs.close()
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_infos
