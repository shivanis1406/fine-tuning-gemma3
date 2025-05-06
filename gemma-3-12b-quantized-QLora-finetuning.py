import os
from huggingface_hub import login
from dotenv import load_dotenv
import json
from sklearn.model_selection import train_test_split

load_dotenv()
hf_token = os.getenv("HF_Token")
login(hf_token)

# Load JSON data
with open("conversations_combined_dataset.json", "r") as f:
    raw_data = json.load(f)

print(f"Total entries: {len(raw_data)}")

def preprocess_conversations(data):
    formatted_data = []
    for example in data:
        conv = example.get("conversations", [])
        if len(conv) >= 2:
            for i in range(0, len(conv) - 1, 2):
                user = conv[i]
                assistant = conv[i + 1]
                if user["role"] == "user" and assistant["role"] == "assistant":
                    formatted_data.append([user, assistant])
    return formatted_data

formatted_data = preprocess_conversations(raw_data)
train_data, test_data = train_test_split(formatted_data, test_size=0.2, random_state=42)

print("Training Data")
print(train_data[0])

print("Testing Data")
print(test_data[0])
# Save the splits
#with open("train_data.json", "w") as f:
#    json.dump(train_data, f, indent=2)

#with open("test_data.json", "w") as f:
#   json.dump(test_data, f, indent=2)

print(f"Train set: {len(train_data)} samples")
print(f"Test set: {len(test_data)} samples")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForImageTextToText, BitsAndBytesConfig

# Hugging Face model id
model_id = "google/gemma-3-12b-it" # or `google/gemma-3-4b-pt`, `google/gemma-3-12b-pt`, `google/gemma-3-27b-pt`

# Select model class based on id
if model_id == "google/gemma-3-12b-it":
    model_class = AutoModelForCausalLM
else:
    model_class = AutoModelForImageTextToText

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    attn_implementation="eager", # Use "flash_attention_2" when running on Ampere or newer GPU
    torch_dtype=torch_dtype, # What torch dtype to use, defaults to auto
    device_map="auto", # Let torch decide how to load the model
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
    bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
)

# Load model and tokenizer
model = model_class.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it") # Load the Instruction Tokenizer to use the official Gemma template


from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=16,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"] # make sure to save the lm_head and embed_tokens as you train the special tokens
)

from trl import SFTConfig

args = SFTConfig(
    output_dir="output",         # directory to save and repository id
    max_seq_length=512,                     # max sequence length for model and packing of the dataset
    packing=True,                           # Groups multiple samples in the dataset into a single sequence
    num_train_epochs=3,                     # number of training epochs
    per_device_train_batch_size=1,          # batch size per device during training
    gradient_accumulation_steps=4,          # number of steps before performing a backward/update pass
    gradient_checkpointing=True,            # use gradient checkpointing to save memory
    optim="adamw_torch_fused",              # use fused adamw optimizer
    logging_steps=10,                       # log every 10 steps
    save_strategy="epoch",                  # save checkpoint every epoch
    learning_rate=2e-4,                     # learning rate, based on QLoRA paper
    fp16=True if torch_dtype == torch.float16 else False,   # use float16 precision
    bf16=True if torch_dtype == torch.bfloat16 else False,   # use bfloat16 precision
    max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,                      # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",           # use constant learning rate scheduler
    push_to_hub=True,                       # push model to hub
    report_to="tensorboard",                # report metrics to tensorboard
    dataset_kwargs={
        "add_special_tokens": False, # We template with special tokens
        "append_concat_token": True, # Add EOS token as separator token between examples
    }
)

from trl import SFTTrainer

# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,
    processing_class=tokenizer
)

# Start training, the model will be automatically saved to the Hub and the output directory
trainer.train()

# Save the final model again to the Hugging Face Hub
trainer.save_model()

# free the memory again
del model
del trainer
torch.cuda.empty_cache()

from peft import PeftModel

# Load Model base model
model = model_class.from_pretrained(model_id, low_cpu_mem_usage=True)

# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, args.output_dir)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")

processor = AutoTokenizer.from_pretrained(args.output_dir)
processor.save_pretrained("merged_model")