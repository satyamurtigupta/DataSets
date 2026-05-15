# =============================================================================
# UPSC Gemma 2 — Stage 2: Instruction Fine-Tuning (SFT)
# =============================================================================
# Run AFTER Stage 1 is complete. Loads the Stage 1 LoRA adapter as the base.
# Teaches the model to answer UPSC questions in chat format.
# =============================================================================


# ── CELL 1 : Install (same as Stage 1 — skip if already done) ─────────────────
!pip install -q \
    transformers==4.44.2 \
    trl==0.9.6 \
    peft==0.12.0 \
    bitsandbytes==0.43.3 \
    accelerate==0.33.0 \
    datasets==2.20.0 \
    huggingface_hub \
    sentencepiece \
    protobuf


# ── CELL 2 : Paths ────────────────────────────────────────────────────────────

from google.colab import drive
drive.mount('/content/drive')

DRIVE_BASE      = "/content/drive/MyDrive/UPSC_SLM"
SFT_JSONL       = f"{DRIVE_BASE}/unified_sft.jsonl"
STAGE1_ADAPTER  = f"{DRIVE_BASE}/gemma2_stage1/lora_adapter_stage1"
OUTPUT_DIR      = f"{DRIVE_BASE}/gemma2_stage2"
HF_MODEL_ID     = "google/gemma-2-2b"
HF_TOKEN        = ""   # your HuggingFace token

import os
assert os.path.exists(SFT_JSONL),      f"SFT dataset not found: {SFT_JSONL}"
assert os.path.exists(STAGE1_ADAPTER), f"Stage 1 adapter not found: {STAGE1_ADAPTER}"
print("✓ All paths OK")


# ── CELL 3 : HF login ─────────────────────────────────────────────────────────

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)


# ── CELL 4 : Load SFT dataset ─────────────────────────────────────────────────

from datasets import load_dataset
from collections import Counter

sft_dataset = load_dataset("json", data_files={"train": SFT_JSONL}, split="train")

print(f"SFT chunks     : {len(sft_dataset):,}")
print(f"Columns        : {sft_dataset.column_names}")

# Verify chat format is present
sample = sft_dataset[0]["text"]
print(f"\nFormat check (first 300 chars):")
print(sample[:300])
print("...")

# The SFT text already contains Gemma 2 chat tokens:
# <start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n...<end_of_turn>
assert "<start_of_turn>" in sample, "SFT records must contain Gemma 2 chat tokens!"
print("\n✓ Chat format verified")

subj_counts = Counter(sft_dataset["subject"])
print("\nChunks by subject:")
for subj, cnt in sorted(subj_counts.items(), key=lambda x: -x[1]):
    print(f"  {subj:<20} {cnt:>5,}")


# ── CELL 5 : Load Stage 1 model + add Stage 2 LoRA ───────────────────────────
# Strategy: load base model → attach Stage 1 adapter → merge → add Stage 2 LoRA
# This way Stage 2 LoRA learns delta on top of the domain-adapted model.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import (
    LoraConfig, get_peft_model, prepare_model_for_kbit_training,
    PeftModel,
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    token=HF_TOKEN,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Attach Stage 1 adapter (domain knowledge)
model = PeftModel.from_pretrained(base_model, STAGE1_ADAPTER, is_trainable=False)

# Prepare for new LoRA training
model = prepare_model_for_kbit_training(model)

# Add fresh Stage 2 LoRA on top
# Lower rank than Stage 1 — SFT is a smaller delta (behaviour, not knowledge)
lora_config_s2 = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model.add_adapter(lora_config_s2, adapter_name="stage2")
model.set_adapter("stage2")
model.print_trainable_parameters()


# ── CELL 6 : Response template for loss masking ───────────────────────────────
# Only compute loss on the MODEL's response tokens, not on the USER prompt.
# This is what separates SFT from CPT.

from trl import DataCollatorForCompletionOnlyLM

# Gemma 2 model-turn start token sequence
RESPONSE_TEMPLATE = "<start_of_turn>model\n"
response_template_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)
print(f"Response template token IDs: {response_template_ids}")

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template_ids,
    tokenizer=tokenizer,
)


# ── CELL 7 : Training arguments ───────────────────────────────────────────────

from transformers import TrainingArguments

MAX_SEQ_LEN = 2048
EPOCHS      = 2        # SFT typically needs 1–3 epochs
BATCH_SIZE  = 2
GRAD_ACCUM  = 8        # effective batch = 16

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,

    optim="paged_adamw_8bit",
    learning_rate=1e-4,             # lower LR than Stage 1 (fine behaviour delta)
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,

    fp16=False,
    bf16=True,

    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,
    report_to="none",

    dataloader_num_workers=2,
    group_by_length=True,
    seed=42,
)


# ── CELL 8 : Train ────────────────────────────────────────────────────────────

from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=sft_dataset,
    args=training_args,

    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LEN,
    packing=False,              # DO NOT pack for SFT — each sample is one conversation
                                # packing would mix user/model turns across chunks
    data_collator=collator,     # masks loss on prompt tokens → only learn from answers
)

print("Starting Stage 2 SFT training...")
print(f"Data  : {len(sft_dataset):,} instruction samples")
print(f"Loss  : only on model response tokens (prompt is masked)")
print("-" * 60)

trainer.train()
print("Stage 2 training complete ✓")


# ── CELL 9 : Save Stage 2 adapter ─────────────────────────────────────────────

ADAPTER_SAVE_PATH_S2 = f"{OUTPUT_DIR}/lora_adapter_stage2"
trainer.model.save_pretrained(ADAPTER_SAVE_PATH_S2)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH_S2)
print(f"Stage 2 LoRA adapter saved → {ADAPTER_SAVE_PATH_S2}")


# ── CELL 10 : Chat inference test ─────────────────────────────────────────────
# Test the full pipeline: domain knowledge (Stage 1) + Q&A behaviour (Stage 2)

model.eval()

def chat(question: str, max_new_tokens: int = 200) -> str:
    """Run the trained model on a UPSC-style question."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,   # adds <start_of_turn>model\n at the end
    )
    inputs = tokenizer(prompt, return_tensors="pt",
                       add_special_tokens=False).to("cuda")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
        )
    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


questions = [
    "What were the main causes of the Revolt of 1857?",
    "Explain the significance of Article 32 of the Indian Constitution.",
    "What is the difference between a monsoon and a cyclone?",
    "Describe the features of the Indus Valley Civilisation.",
    "What is the role of NABARD in Indian agriculture?",
]

print("=" * 60)
print("Chat inference test — Stage 2 fine-tuned model")
print("=" * 60)
for q in questions:
    print(f"\nQ: {q}")
    print(f"A: {chat(q)}")
    print("-" * 60)
