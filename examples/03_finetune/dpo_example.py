"""
examples/03_finetune/dpo_example.py

DPO (Direct Preference Optimization) training script.
Maps to docs/06-alignment-rlhf/README.md "DPO — preference pairs directly, no PPO loop".

NOTE: Install dependencies before running:
    pip install transformers peft trl datasets accelerate bitsandbytes
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
from datasets import Dataset


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # start from SFT model in production
OUTPUT_DIR = "./dpo-output"


# Sample preference data — replace with your own
# Format: prompt, chosen (preferred response), rejected (less preferred)
SAMPLE_PREFERENCE_DATA = [
    {
        "prompt": "Explain LoRA",
        "chosen": "LoRA (Low-Rank Adaptation) freezes the base model and trains small low-rank matrices added to the attention and MLP weights. This makes fine-tuning much cheaper in memory and compute.",
        "rejected": "LoRA is some kind of model thing.",
    },
    {
        "prompt": "What is QLoRA?",
        "chosen": "QLoRA combines 4-bit quantization of the base model with LoRA adapters, enabling fine-tuning of 65B models on a single 48GB GPU.",
        "rejected": "QLoRA uses 8-bit and is faster than LoRA.",
    },
]


def build_dataset() -> Dataset:
    return Dataset.from_list(SAMPLE_PREFERENCE_DATA)


def main():
    # 1. Load model (4-bit for memory efficiency)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA config for DPO
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # 4. DPO config
    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        beta=0.1,  # KL penalty — higher = stay closer to reference
        max_prompt_length=512,
        max_length=1024,
    )

    # 5. Trainer
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL auto-creates from base model
        args=dpo_args,
        train_dataset=build_dataset(),
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
    print(f"\nDPO adapter saved to {OUTPUT_DIR}/final_adapter")


if __name__ == "__main__":
    main()