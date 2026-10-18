#!/bin/bash
# TrajEvolve Trainer for Webshop
# This script runs the TrajEvolve algorithm which extends GRPO with trajectory evolution

set -x

# Environment setup
export VLLM_USE_V1=0
export VLLM_ATTENTION_BACKEND=XFORMERS

export CUDA_VISIBLE_DEVICES='0,1,2,3'
ENGINE=${1:-vllm}

# Training parameters
num_cpus_per_env_worker=0.4
train_data_size=16
val_data_size=128
group_size=8

# Model paths
experiment_name='traj_evolve_qwen2.5_1.5b_w_grpo'
MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct
CKPTS_DIR=$HOME/checkpoints/trajevolve/webshop/${experiment_name}
DISCRIMINATOR_OUTPUT_DIR=$HOME/trajevolve/discriminator_output/webshop
DISCRIMINATOR_MODEL_PATH=$HOME/trajevolve/discriminator_model/webshop
# TrajEvolve specific parameters (configurable)
# Reward formula for success: base + alpha * efficiency + beta * efficiency^n / necessary_steps
# - When redundant steps exist: efficiency term (alpha) dominates
# - When efficiency = 1.0: brevity term (beta/N) activates

# Run TrajEvolve trainer
python3 -u -m verl.trainer.main_traj_evolve \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Webshop \
    env.seed=0 \
    env.max_steps=20 \
    env.rollout.n=$group_size \
    env.history_length=5 \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    traj_evolve.alpha=1.0 \
    traj_evolve.evolve_freq=5 \
    traj_evolve.discriminator_model_path=$DISCRIMINATOR_MODEL_PATH \
    traj_evolve.discriminator_output_dir=$DISCRIMINATOR_OUTPUT_DIR \
    traj_evolve.discriminator_lr=1e-5 \
    traj_evolve.discriminator_epochs=1 \
    traj_evolve.step_credit_value=0.5 \
    +traj_evolve.discriminator_max_length=4096 \
    +traj_evolve.discriminator_micro_batch_size=2 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='trajevolve_webshop' \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${CKPTS_DIR}" $@
