# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
ALFWorld Inference Script
This script loads alfworld data and performs inference using a specified model.
It reuses the GPU distribution, data loading, and inference logic from main_ppo.py.
"""

import os
import json
import hydra
import ray
from pathlib import Path
from datetime import datetime

os.environ['REQUESTS_CA_BUNDLE'] = ''


@hydra.main(config_path="config", config_name="inference_alfworld", version_base=None)
def main(config):
    run_inference(config)


def run_inference(config) -> None:
    if not ray.is_initialized():
        # Initialize ray with the same configuration as main_ppo.py
        print("Initializing Ray...")
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        },
            num_cpus=config.ray_init.num_cpus,
        )

    runner = InferenceRunner.remote()
    results = ray.get(runner.run.remote(config))
    print(f"Inference completed. Results saved to: {results['output_path']}")


@ray.remote(num_cpus=1)
class InferenceRunner:
    def run(self, config):
        """Main inference execution"""
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local

        # Print configuration
        print("=" * 80)
        print("Inference Configuration:")
        print("=" * 80)
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download the checkpoint from hdfs if needed
        local_path = copy_to_local(
            config.model.path, 
            use_shm=config.model.get("use_shm", False)
        )
        print(f"Model loaded from: {local_path}")

        # Create environments using the same method as main_ppo.py
        from agent_system.environments import make_envs
        print("Creating ALFWorld environments...")
        envs, val_envs = make_envs(config)
        
        # Use training envs for inference
        inference_envs = envs
        print("Using training environments")

        # Instantiate tokenizer - same as main_ppo.py
        from verl.utils import hf_processor, hf_tokenizer
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        print("Tokenizer and processor loaded")

        # Define worker classes - same as main_ppo.py
        # 定义 worker 类
        if config.model.strategy in ["fsdp", "fsdp2"]:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.single_controller.ray.base import RayClassWithInitArgs  # 新增：需要导入这个
            from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
            
            actor_rollout_cls = AsyncActorRolloutRefWorker if config.rollout.mode == "async" else ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
        elif config.model.strategy == "megatron":
            from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
            from verl.single_controller.ray.base import RayClassWithInitArgs  # 新增：需要导入这个
            from verl.workers.megatron_workers import ActorRolloutRefWorker
            
            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = NVMegatronRayWorkerGroup
        else:
            raise NotImplementedError(f"Strategy {config.model.strategy} not supported")

        # 创建资源池管理器
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.inference.n_gpus_per_node] * config.inference.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )

        # 关键步骤1：创建资源池
        resource_pool_manager.create_resource_pool()

        # 关键步骤2：获取实际的资源池对象
        resource_pool = resource_pool_manager.get_resource_pool(Role.ActorRollout)

        # 关键步骤3：用 RayClassWithInitArgs 包装 worker 类
        # 构建完整的 actor_rollout_ref 配置用于推理
        from omegaconf import OmegaConf
        actor_rollout_ref_config = OmegaConf.create({
            'model': config.model,
            'rollout': config.rollout,
            'hybrid_engine': True,
            'actor': {
                'strategy': config.model.strategy,
                'ppo_mini_batch_size': 256,
                'ppo_micro_batch_size': None,  # deprecated, use ppo_micro_batch_size_per_gpu
                'ppo_micro_batch_size_per_gpu': 16,
                'use_dynamic_bsz': False,
                'ppo_max_token_len_per_gpu': 16384,
                'grad_clip': 1.0,
                'clip_ratio': 0.2,
                'clip_ratio_low': 0.2,
                'clip_ratio_high': 0.2,
                'clip_ratio_c': 3.0,
                'loss_agg_mode': 'token-mean',
                'entropy_coeff': 0.001,
                'use_invalid_action_penalty': True,
                'invalid_action_penalty_coef': 0.1,
                'use_kl_loss': False,
                'use_torch_compile': False,
                'ppo_epochs': 1,
                'shuffle': False,
                'ulysses_sequence_parallel_size': 1,
                'optim': {
                    'lr': 1e-6,
                    'lr_warmup_steps': -1,
                    'lr_warmup_steps_ratio': 0.0,
                    'min_lr_ratio': 0.0,
                    'num_cycles': 0.5,
                    'warmup_style': 'constant',
                    'total_training_steps': 1,  # dummy value for inference
                    'weight_decay': 0.01,
                },
                'checkpoint': {
                    'contents': ['model', 'optimizer', 'extra'],  # Required by FSDPCheckpointManager
                },
                'fsdp_config': {
                    'wrap_policy': {'min_num_params': 0},
                    'param_offload': False,
                    'optimizer_offload': False,
                    'offload_policy': False,
                    'reshard_after_forward': True,
                    'fsdp_size': -1,
                },
            },
            'ref': {
                'strategy': config.model.strategy,
                'fsdp_config': {
                    'param_offload': False,
                    'reshard_after_forward': True,
                    'wrap_policy': {'min_num_params': 0},
                },
                'log_prob_micro_batch_size_per_gpu': config.rollout.get('log_prob_micro_batch_size_per_gpu', 32),
                'log_prob_use_dynamic_bsz': False,
                'log_prob_max_token_len_per_gpu': 16384,
                'ulysses_sequence_parallel_size': 1,
            },
        })
        
        # 重要：先用 ray.remote() 装饰 worker 类
        remote_actor_cls = ray.remote(actor_rollout_cls)
        
        wrapped_actor_rollout_cls = RayClassWithInitArgs(
            cls=remote_actor_cls,
            config=actor_rollout_ref_config,
            role="actor_rollout",
        )

        # 创建轨迹收集器
        from agent_system.multi_turn_rollout import TrajectoryCollector
        traj_collector = TrajectoryCollector(
            config=config, 
            tokenizer=tokenizer, 
            processor=processor
        )
        print("Trajectory collector initialized")

        # 加载数据集（固定使用训练集）
        from verl.utils.dataset.rl_dataset import collate_fn
        dataset = self._create_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor
        )
        print(f"Training dataset loaded: {len(dataset)} samples")

        # 关键步骤4：使用正确的参数初始化 worker group
        print("Initializing workers...")
        actor_rollout_wg = ray_worker_group_cls(
            resource_pool=resource_pool,              # ✅ 传递 resource_pool 对象
            ray_cls_with_init=wrapped_actor_rollout_cls,  # ✅ 传递包装后的类
            device_name="cuda"
        )
        
        # Initialize actor rollout workers
        self._init_workers(actor_rollout_wg, config, local_path, tokenizer)
        print("Workers initialized successfully")
        
        # Store tokenizer for decoding
        self.tokenizer = tokenizer

        # Run inference
        print("=" * 80)
        print(f"Starting inference with {config.inference.num_rollouts} rollout(s) per task...")
        print("=" * 80)
        
        results = self._run_rollouts(
            dataset=dataset,
            actor_rollout_wg=actor_rollout_wg,
            inference_envs=inference_envs,
            traj_collector=traj_collector,
            collate_fn=collate_fn,
            config=config
        )
        
        # Save results
        output_path = self._save_results(results, config)
        
        print("=" * 80)
        print("Inference Summary:")
        print(f"Total tasks: {len(results)}")
        print(f"Output saved to: {output_path}")
        print("=" * 80)
        
        return {"output_path": output_path, "num_tasks": len(results)}

    def _create_dataset(self, data_paths, data_config, tokenizer, processor):
        """Create dataset - same implementation as main_ppo.py"""
        from torch.utils.data import Dataset
        from verl.utils.dataset.rl_dataset import RLHFDataset

        if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
            from verl.utils.import_utils import load_extern_type
            dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
            if not issubclass(dataset_cls, Dataset):
                raise TypeError(f"The custom dataset class must inherit from torch.utils.data.Dataset")
        else:
            dataset_cls = RLHFDataset

        dataset = dataset_cls(
            data_files=data_paths,
            tokenizer=tokenizer,
            processor=processor,
            config=data_config,
        )
        return dataset

    def _init_workers(self, actor_rollout_wg, config, local_path, tokenizer):
        """Initialize workers with model - similar to RayPPOTrainer"""
        # Initialize workers - config is already passed through RayClassWithInitArgs
        actor_rollout_wg.init_model()
        
        # Set up rollout generation if applicable
        # Note: For VLLM-based rollout, generation config is handled internally

    def _run_rollouts(self, dataset, actor_rollout_wg, inference_envs, traj_collector, collate_fn, config):
        """Run inference rollouts on the dataset"""
        import torch
        from torch.utils.data import DataLoader
        from verl import DataProto
        
        results = []
        
        # Create dataloader (使用训练集的batch_size)
        dataloader = DataLoader(
            dataset,
            batch_size=config.data.train_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        print(f"DataLoader created: {len(dataloader)} batches, batch_size={config.data.train_batch_size}")
        
        for batch_idx, batch in enumerate(dataloader):
            print(f"\nProcessing batch {batch_idx + 1}/{len(dataloader)}...")
            
            # Debug: check batch keys
            if batch_idx == 0:
                print(f"Batch keys: {batch.keys()}")
            
            # Ensure raw_prompt exists in batch (required by trajectory collector)
            if 'raw_prompt' not in batch and 'prompt' in batch:
                # If raw_prompt doesn't exist, create it from prompt
                batch['raw_prompt'] = batch['prompt']
            
            # Separate tensors and non-tensors for DataProto
            tensors = {}
            non_tensors = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    tensors[key] = value
                else:
                    non_tensors[key] = value
            
            # Convert batch to DataProto
            gen_batch = DataProto.from_dict(tensors=tensors, non_tensors=non_tensors)
            
            # Run rollouts for each task
            for rollout_idx in range(config.inference.num_rollouts):
                print(f"  Rollout {rollout_idx + 1}/{config.inference.num_rollouts}...")
                
                # Perform multi-turn rollout - same as training
                gen_batch_output = traj_collector.multi_turn_loop(
                    gen_batch=gen_batch,
                    actor_rollout_wg=actor_rollout_wg,
                    envs=inference_envs,
                    is_train=False,
                )
                
                # Extract results
                batch_results = self._extract_batch_results(gen_batch_output, batch_idx, rollout_idx)
                results.extend(batch_results)
        
        return results

    def _extract_batch_results(self, gen_batch_output, batch_idx, rollout_idx):
        """Extract results from generated batch"""
        import torch
        
        batch_results = []
        batch_size = len(gen_batch_output.non_tensor_batch['raw_prompt'])
        
        for i in range(batch_size):
            result = {
                'batch_idx': batch_idx,
                'rollout_idx': rollout_idx,
                'sample_idx': i,
                'raw_prompt': gen_batch_output.non_tensor_batch['raw_prompt'][i],
                'data_source': gen_batch_output.non_tensor_batch.get('data_source', [None] * batch_size)[i],
            }
            
            # Extract episode information if available
            if 'episode_reward' in gen_batch_output.non_tensor_batch:
                result['episode_reward'] = float(gen_batch_output.non_tensor_batch['episode_reward'][i])
            
            if 'episode_length' in gen_batch_output.non_tensor_batch:
                result['episode_length'] = int(gen_batch_output.non_tensor_batch['episode_length'][i])
            
            if 'success' in gen_batch_output.non_tensor_batch:
                success_data = gen_batch_output.non_tensor_batch['success']
                if isinstance(success_data, dict):
                    result['success'] = {k: bool(v[i]) if hasattr(v, '__getitem__') else bool(v) 
                                       for k, v in success_data.items()}
                else:
                    result['success'] = bool(success_data[i]) if hasattr(success_data, '__getitem__') else bool(success_data)
            
            # Extract trajectory tokens and decode to text
            if 'input_ids' in gen_batch_output.batch:
                # Get the generated sequence
                input_ids = gen_batch_output.batch['input_ids'][i]
                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.cpu().numpy()
                result['generated_ids'] = input_ids.tolist()
                
                # Decode to text
                try:
                    result['generated_text'] = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                    result['generated_text_clean'] = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                except Exception as e:
                    print(f"Warning: Failed to decode tokens for sample {i}: {e}")
                    result['generated_text'] = None
                    result['generated_text_clean'] = None
            
            # Extract step-by-step trajectory if available
            if 'trajectory' in gen_batch_output.non_tensor_batch:
                trajectory_data = gen_batch_output.non_tensor_batch['trajectory']
                if isinstance(trajectory_data, (list, tuple)) and len(trajectory_data) > i:
                    result['trajectory_steps'] = trajectory_data[i]
            
            # Extract multi-turn information if available
            if 'responses' in gen_batch_output.non_tensor_batch:
                responses = gen_batch_output.non_tensor_batch['responses']
                if isinstance(responses, (list, tuple)) and len(responses) > i:
                    result['step_by_step_responses'] = responses[i]
            
            if 'observations' in gen_batch_output.non_tensor_batch:
                observations = gen_batch_output.non_tensor_batch['observations']
                if isinstance(observations, (list, tuple)) and len(observations) > i:
                    result['step_by_step_observations'] = observations[i]
            
            if 'actions' in gen_batch_output.non_tensor_batch:
                actions = gen_batch_output.non_tensor_batch['actions']
                if isinstance(actions, (list, tuple)) and len(actions) > i:
                    result['step_by_step_actions'] = actions[i]
            
            if 'rewards' in gen_batch_output.non_tensor_batch:
                rewards = gen_batch_output.non_tensor_batch['rewards']
                if isinstance(rewards, (list, tuple)) and len(rewards) > i:
                    result['step_by_step_rewards'] = [float(r) for r in rewards[i]]
            
            batch_results.append(result)
        
        return batch_results

    def _save_results(self, results, config):
        """Save inference results to file"""
        import json
        from datetime import datetime
        
        # Create output directory
        output_dir = Path(config.inference.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(config.model.path).name
        output_filename = f"inference_{model_name}_{timestamp}.json"
        output_path = output_dir / output_filename
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': {
                    'model_path': config.model.path,
                    'num_rollouts': config.inference.num_rollouts,
                    'dataset': 'train',  # 固定使用训练集
                    'timestamp': timestamp,
                },
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        # Also save a summary
        summary_path = output_dir / f"summary_{model_name}_{timestamp}.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Inference Summary\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Model: {config.model.path}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total tasks: {len(results)}\n")
            f.write(f"Rollouts per task: {config.inference.num_rollouts}\n")
            
            # Calculate success rate if available
            success_results = [r for r in results if 'success' in r]
            if success_results:
                success_count = sum(1 for r in success_results 
                                  if (isinstance(r['success'], dict) and any(r['success'].values())) or
                                     (isinstance(r['success'], bool) and r['success']))
                f.write(f"Success rate: {success_count}/{len(success_results)} ({100*success_count/len(success_results):.2f}%)\n")
            
            # Calculate average reward if available
            reward_results = [r['episode_reward'] for r in results if 'episode_reward' in r]
            if reward_results:
                avg_reward = sum(reward_results) / len(reward_results)
                f.write(f"Average reward: {avg_reward:.4f}\n")
        
        return str(output_path)


if __name__ == "__main__":
    main()

