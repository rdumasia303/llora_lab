#!/usr/bin/env python3
"""
Llora Lab Training Script

This script handles the fine-tuning process to create LoRA adapters for LLMs.
It uses Unsloth for efficient training and supports various configuration options.
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def setup_logging(log_level: str, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with the specified level and optional file output."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Configure root logger
    handlers = []
    
    # Always log to stdout
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    handlers.append(stream_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Create and return logger for this module
    logger = logging.getLogger("llm_studio.train")
    return logger


def setup_model_and_tokenizer(
    model_name: str, 
    max_seq_length: int,
    use_nested_quant: bool = False,
    logger: Optional[logging.Logger] = None
) -> tuple:
    """Initialize the model and tokenizer."""
    if logger:
        logger.info(f"Loading model {model_name} with max sequence length {max_seq_length}")
    
    # Determine if we should use bfloat16
    use_bf16 = FastLanguageModel.is_bfloat16_supported()
    if logger:
        logger.info(f"Using bfloat16: {use_bf16}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Will be automatically determined
        load_in_4bit=True,
        use_nested_quant=use_nested_quant
    )
    
    return model, tokenizer


def setup_peft_model(
    model, 
    max_seq_length: int, 
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    use_rslora: bool = False,
    logger: Optional[logging.Logger] = None
) -> Any:
    """Configure the model for PEFT (Parameter-Efficient Fine-Tuning)."""
    if logger:
        logger.info(f"Setting up PEFT model with rank {lora_rank}, alpha {lora_alpha}")
    
    # Use default target modules if none provided
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return FastLanguageModel.get_peft_model(
        model, 
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=use_rslora,
        loftq_config=None,
    )


def load_training_dataset(dataset_path: str, logger: Optional[logging.Logger] = None) -> Any:
    """Load and prepare the training dataset."""
    if logger:
        logger.info(f"Loading dataset from {dataset_path}")
    
    # Check file extension
    if dataset_path.endswith('.jsonl'):
        dataset = load_dataset('json', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.csv'):
        dataset = load_dataset('csv', data_files=dataset_path, split='train')
    elif dataset_path.endswith('.txt'):
        # For text files, we'll load line by line and format as needed
        dataset = load_dataset('text', data_files=dataset_path, split='train')
        # Convert to expected format with 'text' field
        dataset = dataset.map(lambda example: {"text": example["text"]})
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_path}")
    
    if logger:
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Analyze dataset structure
        first_example = dataset[0]
        logger.info(f"Example fields: {list(first_example.keys())}")
        
        # Check if there's a 'text' field as required
        if 'text' not in first_example:
            logger.warning("Dataset does not contain 'text' field which is required for training")
    
    return dataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LoRA adapter')
    
    # Basic configuration
    parser.add_argument('--model-name', required=True, help='Base model to fine-tune')
    parser.add_argument('--dataset', required=True, help='Training dataset file')
    parser.add_argument('--output-dir', required=True, help='Output directory for adapter')
    
    # LoRA parameters
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=2, help='Per-device batch size')
    parser.add_argument('--gradient-accumulation', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--warmup-steps', type=int, default=10, help='Warmup steps')
    parser.add_argument('--steps', type=int, default=60, help='Total training steps')
    parser.add_argument('--max-seq-length', type=int, default=8192, help='Maximum sequence length')
    parser.add_argument('--seed', type=int, default=3407, help='Random seed')
    
    # Advanced options
    parser.add_argument('--use-peft', action='store_true', help='Use PEFT for training')
    parser.add_argument('--use-nested-quant', action='store_true', help='Use nested quantization')
    parser.add_argument('--chat-template', default="llama-3.1", help='Chat template to use')
    parser.add_argument('--dataset-text-field', default="text", help='Dataset field containing training text')
    
    # Logging options
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], 
                        help='Logging level')
    parser.add_argument('--log-file', help='Log file path')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    logger.info("Starting training process")
    logger.info(f"Arguments: {args}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup model and tokenizer
    max_seq_length = args.max_seq_length
    model, tokenizer = setup_model_and_tokenizer(
        args.model_name, 
        max_seq_length,
        use_nested_quant=args.use_nested_quant,
        logger=logger
    )
    
    # Apply chat template
    logger.info(f"Applying chat template: {args.chat_template}")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=args.chat_template,
    )
    
    # Setup PEFT model
    model = setup_peft_model(
        model, 
        max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        logger=logger
    )
    
    # Load dataset
    dataset = load_training_dataset(args.dataset, logger=logger)
    
    # Save training metadata
    metadata = {
        "model_name": args.model_name,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
        "chat_template": args.chat_template,
        "training_parameters": {
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "seed": args.seed
        },
        "dataset": {
            "path": args.dataset,
            "samples": len(dataset)
        }
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Configure trainer
    logger.info("Configuring trainer")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field=args.dataset_text_field,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            warmup_steps=args.warmup_steps,
            max_steps=args.steps,
            learning_rate=args.learning_rate,
            fp16=not FastLanguageModel.is_bfloat16_supported(),
            bf16=FastLanguageModel.is_bfloat16_supported(),
            logging_steps=1,
            output_dir=args.output_dir,
            optim="adamw_8bit",
            seed=args.seed,
        ),
    )
    
    # Train and save
    logger.info("Starting training")
    trainer.train()
    
    logger.info("Training complete, saving model")
    model.save_pretrained(args.output_dir, save_method="merged_4bit")
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
