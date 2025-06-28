"""
Training script for Feynomenon AI.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- Configuration ---
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3" # Or v0.2 if v0.3 is not public on HF Hub yet
DATASET_NAME = "HuggingFaceH4/ultrachat_200k" # Example dataset, replace with your tutor dataset
OUTPUT_DIR = "./mistral-feynman-tutor"
GRADIENT_ACCUMULATION_STEPS = 4 # Adjust based on GPU memory and batch size

# Hugging Face Hub Token (for gated models like Mistral)
# Ensure you have logged in: huggingface-cli login
# Or set HF_TOKEN environment variable
HF_TOKEN = os.environ.get("HF_TOKEN", "your_huggingface_read_token_if_needed")

# --- QLoRA Configuration ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# --- Load Model and Tokenizer ---
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16, # Use bfloat16 for better numerical stability with A100s
    device_map="auto", # Automatically maps model to available devices (GPUs)
    token=HF_TOKEN # Pass token if necessary for gated models
)
model.config.use_cache = False # Recommended for fine-tuning
model = prepare_model_for_kbit_training(model) # Prepare for QLoRA
model = get_peft_model(model, lora_config)
print("Model loaded and prepared for QLoRA training.")
model.print_trainable_parameters() # See how many parameters are actually trainable

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token # Or specify a different pad token if needed
tokenizer.padding_side = "right" # Important for causal LMs

# --- Load and Prepare Dataset (Feynman Technique Data) ---
# Your dataset should contain examples formatted for the AI tutor following Feynman technique.
# For example:
# {"text": "Explain [Concept] to me like I'm 5, then like I'm a high schooler, then like a university student."}
# or
# {"text": "User: Can you explain [Concept] using the Feynman technique?\nAssistant: [Explanation Level 1]\n[Explanation Level 2]\n[Explanation Level 3]"}

print(f"Loading dataset: {DATASET_NAME}...")
# For demo, using a general instruction dataset. Replace with your custom data.
# For actual Feynman tutoring, you'd curate specific examples.
# Example of a custom dataset loading:
# dataset = load_dataset("json", data_files="your_feynman_data.jsonl")
dataset = load_dataset(DATASET_NAME, split="train[:10000]") # Load a subset for demo

# Function to format the dataset for training (SFTTrainer expects 'text' column)
def formatting_prompts_func(example):
    # This is a placeholder. You need to format your specific tutor data here.
    # For ultrachat, it's typically 'messages'
    # For a tutor:
    # return {"text": f"### User: {example['instruction']}\n### Assistant: {example['response']}"}
    
    # For ultrachat, we need to flatten the conversation:
    formatted_text = ""
    for message in example['messages']:
        if message['role'] == 'user':
            formatted_text += f"### User: {message['content']}\n"
        elif message['role'] == 'assistant':
            formatted_text += f"### Assistant: {message['content']}\n"
    return {"text": formatted_text.strip()}

# Apply formatting function to your dataset
dataset = dataset.map(formatting_prompts_func, remove_columns=dataset.column_names)
print("Dataset loaded and formatted.")

# --- Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1, # Adjust this based on your GPU VRAM and gradient_accumulation_steps
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=500, # Or num_train_epochs
    learning_rate=2e-4,
    fp16=False, # Set to True if using NVIDIA V100/T4. Use bf16=True for A100/H100.
    bf16=True, # Use bfloat16 for A100/H100 GPUs
    logging_steps=10,
    save_steps=100,
    eval_steps=100, # If you have a validation set
    save_total_limit=3,
    push_to_hub=False, # Set to True to push to Hugging Face Hub
    report_to="tensorboard", # For logging with TensorBoard
    # You may add evaluation_strategy="steps" if you have a validation split
)

# --- SFTTrainer ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024, # Adjust based on your data and GPU VRAM
    tokenizer=tokenizer,
    args=training_args,
)

# --- Train the Model ---
print("Starting training...")
trainer.train()
print("Training complete.")

# --- Save Model and Tokenizer ---
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model and tokenizer saved to {OUTPUT_DIR}")

# Optional: Push to Hugging Face Hub
# if training_args.push_to_hub:
#     trainer.push_to_hub()
#     tokenizer.push_to_hub()