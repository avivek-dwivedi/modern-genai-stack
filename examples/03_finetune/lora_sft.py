"""
examples/03_finetune/lora_sft.py

SFT with LoRA on a small open-weight LLM.
Maps directly to docs/05-finetuning-peft/README.md "LoRA — the canonical PEFT method".

NOTE: Install dependencies before running:
    pip install transformers peft trl datasets accelerate bitsandbytes
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset


# ---------- Configuration ----------

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"  # small for demo; swap for your model
OUTPUT_DIR = "./lora-output"
MAX_SEQ_LENGTH = 2048


# ---------- Sample data (replace with your own) ----------

SAMPLE_INSTRUCTION_DATA = [
    {"text": "### Instruction: Explain LoRA in one sentence.\n### Response: LoRA adds low-rank trainable matrices to frozen model weights, enabling efficient adaptation."},
    {"text": "### Instruction: What is QLoRA?\n### Response: QLoRA loads the base model in 4-bit (NF4) and trains LoRA adapters on top, allowing 65B models on a single 48GB GPU."},
    {"text": "### Instruction: What does DoRA improve?\n### Response: DoRA decomposes weight into magnitude and direction, applying LoRA to direction only, which improves quality over vanilla LoRA."},
]


def build_dataset() -> Dataset:
    """Build a HuggingFace Dataset from instruction-format text."""
    return Dataset.from_list(SAMPLE_INSTRUCTION_DATA)


def main():
    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Load model in 4-bit (for QLoRA) — comment out for vanilla LoRA
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA config
    lora_config = LoraConfig(
        r=16,                    # rank — higher = more capacity, more memory
        lora_alpha=32,           # scaling — typically 2× rank
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_steps=10,
    )

    # 5. Build dataset
    train_dataset = build_dataset()

    # 6. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
    )

    # 7. Train
    trainer.train()

    # 8. Save adapter
    trainer.model.save_pretrained(f"{OUTPUT_DIR}/final_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_adapter")

    print(f"\nAdapter saved to {OUTPUT_DIR}/final_adapter")


if __name__ == "__main__":
    main()