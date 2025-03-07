#!/usr/bin/env python3
"""
Llora Lab Test Adapter Script

This script loads a trained LoRA adapter and runs inference to test its
performance with a given prompt.
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import torch
from transformers import TextStreamer, GenerationConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def setup_logging(log_level: str) -> logging.Logger:
    """Configure logging with the specified level"""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger("llm_studio.test")


def load_adapter_metadata(adapter_path: str, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Load metadata from the adapter directory"""
    metadata_path = os.path.join(adapter_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        if logger:
            logger.warning(f"No metadata.json found in {adapter_path}, using defaults")
        return {
            "model_name": None,
            "chat_template": "llama-3.1",
            "max_seq_length": 8192
        }
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    if logger:
        logger.info(f"Loaded adapter metadata: {metadata}")
    
    return metadata


def load_model_and_tokenizer(
    adapter_path: str, 
    model_name: Optional[str] = None,
    max_seq_length: int = 8192,
    logger: Optional[logging.Logger] = None
) -> tuple:
    """Load the model with a specific adapter"""
    
    # If model_name is not provided, try to get it from adapter metadata
    if model_name is None:
        metadata = load_adapter_metadata(adapter_path, logger)
        model_name = metadata.get("model_name")
        
        if model_name is None:
            raise ValueError("Model name not provided and not found in adapter metadata")
        
        # Update max_seq_length from metadata if available
        if "max_seq_length" in metadata:
            max_seq_length = metadata["max_seq_length"]
            if logger:
                logger.info(f"Using max_seq_length from metadata: {max_seq_length}")
    
    if logger:
        logger.info(f"Loading model {model_name} with adapter from {adapter_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,  # Load directly from adapter path
        max_seq_length=max_seq_length,
        dtype=None,  # Automatically determined
        load_in_4bit=True
    )
    
    # Set up tokenizer with chat template
    metadata = load_adapter_metadata(adapter_path, logger)
    chat_template = metadata.get("chat_template", "llama-3.1")
    
    if logger:
        logger.info(f"Using chat template: {chat_template}")
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
    )
    
    # Enable faster inference
    if logger:
        logger.info("Optimizing model for inference")
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def generate_response(
    model, 
    tokenizer, 
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
    logger: Optional[logging.Logger] = None
) -> str:
    """Generate a response for a given prompt"""
    if logger:
        logger.info(f"Generating response for prompt: {prompt[:50]}...")
    
    # Format the prompt as chat
    messages = [{"role": "user", "content": prompt}]
    
    # Create the model inputs
    inputs = tokenizer.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to("cuda")
    
    # Set up streamer for real-time output
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    
    # Configure generation parameters
    generation_config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    # Generate response
    if logger:
        logger.info("Starting generation")
    
    output = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        generation_config=generation_config
    )
    
    # Decode the generated text (excluding prompt)
    prompt_length = inputs.shape[1]
    generated_tokens = output[0][prompt_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if logger:
        logger.info(f"Generation complete, response length: {len(response)}")
    
    return response


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test a trained adapter')
    parser.add_argument('--adapter', required=True, help='Path to the adapter directory')
    parser.add_argument('--model-name', help='Base model name (will use metadata if not provided)')
    parser.add_argument('--prompt', required=True, help='Test prompt')
    parser.add_argument('--temperature', type=float, default=0.7, help='Generation temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling parameter')
    parser.add_argument('--max-tokens', type=int, default=256, help='Maximum tokens to generate')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'], 
                       help='Logging level')
    
    return parser.parse_args()


def main():
    """Main function to test an adapter"""
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting adapter testing")
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model_and_tokenizer(
            adapter_path=args.adapter,
            model_name=args.model_name,
            logger=logger
        )
        
        # Print test parameters
        logger.info(f"Prompt: {args.prompt}")
        logger.info(f"Generation parameters: temperature={args.temperature}, "
                    f"top_p={args.top_p}, max_tokens={args.max_tokens}")
        
        # Generate and print response
        print("\nPrompt:", args.prompt)
        print("\nResponse:")
        response = generate_response(
            model, 
            tokenizer, 
            args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            logger=logger
        )
        
        # Save response to file if requested
        if os.environ.get("SAVE_RESPONSE"):
            output_file = os.path.join(args.adapter, "test_response.txt")
            with open(output_file, "w") as f:
                f.write(f"Prompt: {args.prompt}\n\n")
                f.write(f"Response: {response}\n")
            logger.info(f"Response saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

