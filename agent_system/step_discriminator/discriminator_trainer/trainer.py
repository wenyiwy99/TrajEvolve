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
Step-Level Credit Discriminator Training Script.

This script trains a step-level credit discriminator using the verl FSDP SFT Trainer.
It reuses the existing training infrastructure with custom configuration.

Usage:
    torchrun --nproc_per_node=8 train_classifier.py \
        data.train_files=/path/to/train.parquet \
        data.val_files=/path/to/val.parquet \
        model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct
"""

import os
import sys

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from verl.utils import hf_tokenizer
from verl.utils.dataset import SFTDataset
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.device import get_device_name
from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer


def create_discriminator_dataset(data_paths, data_config, tokenizer):
    """
    Create a dataset for step-level credit discriminator training.
    
    Args:
        data_paths: Path(s) to parquet files
        data_config: Data configuration
        tokenizer: Tokenizer instance
        
    Returns:
        SFTDataset instance
    """
    return SFTDataset(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        config=data_config
    )


def evaluate_discriminator(model_path: str, val_parquet_path: str, device: str = "cuda:0", sample_ratio: float = 0.1):
    """
    Evaluate the trained discriminator on validation set.
    
    Args:
        model_path: Path to the trained model
        val_parquet_path: Path to validation parquet file
        device: Device to use for inference
        sample_ratio: Ratio of samples to evaluate (default 0.1 = 10%)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("=" * 60)
    print("Evaluating Discriminator Accuracy")
    print("=" * 60)
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)
    model.eval()
    
    # Load validation data
    val_df = pd.read_parquet(val_parquet_path)
    
    # Sample a subset for evaluation
    if sample_ratio < 1.0:
        val_df = val_df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"Sampled {len(val_df)} samples ({sample_ratio*100:.0f}% of total)")
    
    correct = 0
    total = 0
    necessary_correct = 0
    necessary_total = 0
    redundant_correct = 0
    redundant_total = 0
    
    with torch.no_grad():
        for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
            prompt = row["prompt"]
            true_label = row["response"].strip().lower()
            
            # Apply chat template (same as training)
            prompt_chat = [{"role": "user", "content": prompt}]
            prompt_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False).to(device)
            
            # Generate prediction
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
            # Decode prediction
            generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
            
            # Extract the first word as the label
            pred_label = prediction.split()[0] if prediction else ""
            
            # Check if correct
            is_correct = pred_label == true_label
            if is_correct:
                correct += 1
            total += 1
            
            # Track per-class accuracy
            if true_label == "necessary":
                necessary_total += 1
                if is_correct:
                    necessary_correct += 1
            elif true_label == "redundant":
                redundant_total += 1
                if is_correct:
                    redundant_correct += 1
    
    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    necessary_acc = necessary_correct / necessary_total * 100 if necessary_total > 0 else 0
    redundant_acc = redundant_correct / redundant_total * 100 if redundant_total > 0 else 0
    
    metrics = {
        "accuracy": accuracy,
        "total_samples": total,
        "correct_samples": correct,
        "necessary_accuracy": necessary_acc,
        "necessary_total": necessary_total,
        "necessary_correct": necessary_correct,
        "redundant_accuracy": redundant_acc,
        "redundant_total": redundant_total,
        "redundant_correct": redundant_correct,
    }
    
    print("=" * 60)
    print("Evaluation Results:")
    print(f"  Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"  Necessary Accuracy: {necessary_acc:.2f}% ({necessary_correct}/{necessary_total})")
    print(f"  Redundant Accuracy: {redundant_acc:.2f}% ({redundant_correct}/{redundant_total})")
    print("=" * 60)
    
    return metrics


@hydra.main(config_path="config", config_name="trainer", version_base=None)
def main(config: DictConfig):
    """Main training function."""
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()
    
    # Initialize device mesh
    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",)
    )
    
    # Initialize ulysses device mesh for sequence parallelism
    sp_size = getattr(config, "ulysses_sequence_parallel_size", 1)
    dp_size = world_size // sp_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, sp_size),
        mesh_dim_names=("dp", "sp")
    )
    
    # Load tokenizer
    local_model_path = copy_to_local(
        src=config.model.partial_pretrain,
        verbose=True
    )
    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code
    )
    
    if rank == 0:
        print("=" * 60)
        print("Step Credit Discriminator Training")
        print("=" * 60)
        print(f"Model: {config.model.partial_pretrain}")
        print(f"Train data: {config.data.train_files}")
        print(f"Val data: {config.data.val_files}")
        print(f"Max length: {config.data.max_length}")
        print(f"Total epochs: {config.trainer.total_epochs}")
        print(f"Learning rate: {config.optim.lr}")
        print("=" * 60)
    
    # Create datasets
    train_dataset = create_discriminator_dataset(
        config.data.train_files,
        config.data,
        tokenizer
    )
    val_dataset = create_discriminator_dataset(
        config.data.val_files,
        config.data,
        tokenizer
    )
    
    if rank == 0:
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Val dataset size: {len(val_dataset)}")
    
    # Create trainer and train
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    trainer.fit()
    
    if rank == 0:
        print("=" * 60)
        print("Training complete!")
        print(f"Model saved to: {config.trainer.default_local_dir}")
        print("=" * 60)
        
        # Evaluate the model on training set (10% sample)
        try:
            print("\n[Evaluating on Training Set (10% sample)]")
            train_metrics = evaluate_discriminator(
                model_path=config.trainer.default_local_dir,
                val_parquet_path=config.data.train_files,
                device="cuda:0",
                sample_ratio=0.1
            )
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")
            print("You can manually run evaluation later.")


if __name__ == "__main__":
    main()
