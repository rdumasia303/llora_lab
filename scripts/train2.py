from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse

def setup_model_and_tokenizer(model_name: str, max_seq_length: int):
    """Initialize the model and tokenizer."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )
    
    return model, tokenizer

def setup_peft_model(model, max_seq_length: int):
    """Configure the model for PEFT training."""
    return FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

def main():
    print('Make sure you are pointing to a dataset if you have your own, otherwise will look at dataset.jsonl')
    parser = argparse.ArgumentParser(description='Train LoRA adapter')
    parser.add_argument('--model-name', 
                      default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
                      help='Base model to fine-tune')
    parser.add_argument('--dataset', default="dataset.jsonl",
                      help='Training dataset file')
    parser.add_argument('--output-dir', 
                      default="/workspace/adapters/latest",
                      help='Output directory for model')
    args = parser.parse_args()

    max_seq_length = 8192
    
    # Setup
    model, tokenizer = setup_model_and_tokenizer(args.model_name, max_seq_length)
    model = setup_peft_model(model, max_seq_length)
    
    # Load dataset
    dataset = load_dataset("json", data_files=args.dataset, split='train')
    
    # Configure trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=args.output_dir,
            optim="adamw_8bit",
            seed=3407,
        ),
    )
    
    # Train and save
    trainer.train()
    model.save_pretrained(args.output_dir, save_method = "merged_4bit")
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

