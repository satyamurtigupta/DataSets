# =============================================================================
# UPSC Gemma 2 — Stage 1: Domain Adaptation (Continued Pre-Training)
# =============================================================================
# Copy each section into a separate Colab cell.
# Runtime: GPU → T4 (free) works with 4-bit QLoRA | A100 is faster
# =============================================================================


# ── CELL 1 : Install dependencies ─────────────────────────────────────────────
# Run this first, then Runtime → Restart session

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


# ── CELL 2 : Mount Google Drive & set paths ───────────────────────────────────
# Upload dataset_output_final/combined/unified_pretrain.jsonl to your Drive first.
# Suggested path: MyDrive/UPSC_SLM/unified_pretrain.jsonl

from google.colab import drive
drive.mount('/content/drive')

import os

# ── Edit these paths ──
DRIVE_BASE        = "/content/drive/MyDrive/UPSC_SLM"
PRETRAIN_JSONL    = f"{DRIVE_BASE}/unified_pretrain.jsonl"   # your dataset
OUTPUT_DIR        = f"{DRIVE_BASE}/gemma2_stage1"            # checkpoints saved here
HF_MODEL_ID       = "google/gemma-2-2b"                      # base model
HF_TOKEN          = ""   # paste your HuggingFace token (needs Gemma licence accepted)

# Verify the dataset file exists
assert os.path.exists(PRETRAIN_JSONL), (
    f"Dataset not found at {PRETRAIN_JSONL}\n"
    "Upload unified_pretrain.jsonl to Google Drive first."
)
print(f"✓ Dataset found: {PRETRAIN_JSONL}")


# ── CELL 3 : Login to HuggingFace ────────────────────────────────────────────
# Required to download Gemma 2 (gated model — accept licence at hf.co/google/gemma-2-2b)

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=False)


# ── CELL 4 : Load & inspect dataset ──────────────────────────────────────────

from datasets import load_dataset
import json

# Load the JSONL
raw_dataset = load_dataset("json", data_files={"train": PRETRAIN_JSONL}, split="train")

print(f"Total chunks   : {len(raw_dataset):,}")
print(f"Columns        : {raw_dataset.column_names}")
print(f"Sample subjects: {set(list(raw_dataset['subject'])[:20])}")

# Quick quality check
import statistics
word_counts = raw_dataset["word_count"]
qual_scores = raw_dataset["quality_score"]
print(f"\nWord count  — min={min(word_counts)}  max={max(word_counts)}  "
      f"median={statistics.median(word_counts):.0f}")
print(f"Quality     — min={min(qual_scores):.3f}  "
      f"mean={statistics.mean(qual_scores):.3f}  "
      f"max={max(qual_scores):.3f}")

# Subject distribution
from collections import Counter
subj_counts = Counter(raw_dataset["subject"])
print("\nChunks by subject:")
for subj, cnt in sorted(subj_counts.items(), key=lambda x: -x[1]):
    print(f"  {subj:<20} {cnt:>5,}")


# ── CELL 5 : Configure model + 4-bit QLoRA ────────────────────────────────────

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ── Quantisation config (4-bit NF4 — fits T4 16 GB) ──
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # NF4 is best for LLMs
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,     # extra ~0.4 GB saving
)

# ── Tokenizer ──
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID, token=HF_TOKEN)
tokenizer.padding_side = "right"   # Gemma 2 requirement

# Gemma 2 has no explicit pad token — reuse EOS
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── Base model (4-bit) ──
model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",          # spread across all available GPUs/CPU
    attn_implementation="eager",  # use "flash_attention_2" if on A100
    token=HF_TOKEN,
)
model.config.use_cache = False          # required during training
model.config.pretraining_tp = 1

# ── Prepare for k-bit training (adds gradient checkpointing) ──
model = prepare_model_for_kbit_training(model)

# ── LoRA config ──
# Target all linear projections in attention + MLP (standard for Gemma 2)
lora_config = LoraConfig(
    r=16,                   # rank — higher = more capacity, more VRAM
    lora_alpha=32,          # scaling factor (usually 2×r)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",   # attention
        "gate_proj", "up_proj", "down_proj",        # MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~40–80 M trainable out of ~2 B total (~2–4%)


# ── CELL 6 : Training configuration ──────────────────────────────────────────

from transformers import TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# How many chunks to actually train on
# Start with a subset for a quick sanity-check run, then set to None for full run
TRAIN_SUBSET = None      # None = use all 13,974 chunks
MAX_SEQ_LEN  = 2048      # packing fills sequences to this length
EPOCHS       = 1         # 1 epoch ≈ sees each chunk once; try 2–3 for better adaptation
BATCH_SIZE   = 2         # per-device batch (T4 limit with 4-bit + packing)
GRAD_ACCUM   = 8         # effective batch = BATCH_SIZE × GRAD_ACCUM = 16

train_data = raw_dataset
if TRAIN_SUBSET:
    train_data = raw_dataset.select(range(TRAIN_SUBSET))

print(f"Training on {len(train_data):,} chunks")
print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
steps_per_epoch = len(train_data) // (BATCH_SIZE * GRAD_ACCUM)
print(f"Steps per epoch     : {steps_per_epoch:,}")
print(f"Total steps         : {steps_per_epoch * EPOCHS:,}")


training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    # ── Epochs & steps ──
    num_train_epochs=EPOCHS,

    # ── Batch ──
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,        # saves ~30% VRAM at small speed cost

    # ── Optimiser ──
    optim="paged_adamw_8bit",           # 8-bit Adam — saves VRAM
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,                  # 5% of steps as warmup

    # ── Precision ──
    fp16=False,
    bf16=True,                          # bfloat16 on A100/T4 Ampere+

    # ── Logging & saving ──
    logging_steps=25,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=2,                 # keep only 2 latest checkpoints
    report_to="none",                   # set to "wandb" if you use W&B

    # ── Misc ──
    dataloader_num_workers=2,
    group_by_length=True,               # batch similar-length sequences → less padding
    seed=42,
)


# ── CELL 7 : Build trainer & train ────────────────────────────────────────────

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_data,
    args=training_args,

    # ── Key settings for Stage 1 CPT ──
    dataset_text_field="text",          # column in JSONL that holds the raw text
    max_seq_length=MAX_SEQ_LEN,
    packing=True,                       # packs multiple short chunks into one sequence
                                        # → much higher GPU utilisation
                                        # → EOS tokens are inserted between chunks automatically

    # ── No response template needed for CPT (full next-token prediction) ──
)

print("Starting Stage 1 training...")
print(f"Model : {HF_MODEL_ID}")
print(f"LoRA  : r={lora_config.r}  alpha={lora_config.lora_alpha}")
print(f"Data  : {len(train_data):,} chunks → packed into {MAX_SEQ_LEN}-token sequences")
print("-" * 60)

trainer.train()

print("Training complete ✓")


# ── CELL 8 : Save the LoRA adapter ────────────────────────────────────────────
# Saves only the LoRA weights (~80 MB), not the full 2B model

ADAPTER_SAVE_PATH = f"{OUTPUT_DIR}/lora_adapter_stage1"
trainer.model.save_pretrained(ADAPTER_SAVE_PATH)
tokenizer.save_pretrained(ADAPTER_SAVE_PATH)
print(f"LoRA adapter saved → {ADAPTER_SAVE_PATH}")

# Optional: push to HuggingFace Hub
# trainer.model.push_to_hub("your-hf-username/gemma2-upsc-stage1")
# tokenizer.push_to_hub("your-hf-username/gemma2-upsc-stage1")


# ── CELL 9 : (Optional) Merge LoRA into full model & save ─────────────────────
# Do this only if you need a single merged model file (e.g., for GGUF / Ollama)
# Requires ~12 GB VRAM or CPU offload

from peft import PeftModel
from transformers import AutoModelForCausalLM
import torch

MERGED_SAVE_PATH = f"{OUTPUT_DIR}/merged_stage1"

print("Loading base model in fp16 for merging...")
base_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cpu",       # merge on CPU to avoid VRAM OOM
    token=HF_TOKEN,
)

print("Merging LoRA weights...")
merged = PeftModel.from_pretrained(base_model, ADAPTER_SAVE_PATH)
merged = merged.merge_and_unload()

merged.save_pretrained(MERGED_SAVE_PATH)
tokenizer.save_pretrained(MERGED_SAVE_PATH)
print(f"Merged model saved → {MERGED_SAVE_PATH}")
del base_model, merged
torch.cuda.empty_cache()


# ── CELL 10 : Quick inference test ────────────────────────────────────────────
# Test that the adapted model produces sensible UPSC domain text

from peft import PeftModel
import torch

# Load adapter on top of base (lighter than merged)
test_model = AutoModelForCausalLM.from_pretrained(
    HF_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    token=HF_TOKEN,
)
test_model = PeftModel.from_pretrained(test_model, ADAPTER_SAVE_PATH)
test_model.eval()

prompts = [
    "The Gupta Empire was known for its",
    "Article 370 of the Indian Constitution dealt with",
    "The Chipko Movement was related to",
    "Monetary policy in India is controlled by",
]

print("=" * 60)
print("Inference test — Stage 1 adapted model")
print("=" * 60)

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = test_model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\nPrompt : {prompt}")
    print(f"Output : {response[len(prompt):]}")
    print("-" * 60)
