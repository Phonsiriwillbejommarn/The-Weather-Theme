"""
prune_typhoon_3_5b.py
─────────────────────────────────────────────────────────
Layer Pruning Script: typhoon-ai/typhoon-7b → ~3.5B init
Architecture : Mistral (32 layers → 16 layers, even-skip)
Strategy     : Copy even-indexed teacher layers to student
Output       : ./typhoon-3.5b-init/
─────────────────────────────────────────────────────────
Requirements:
    pip install torch transformers accelerate huggingface_hub
    huggingface-cli login   ← run once to authenticate
"""

import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# ─────────────────────────────────────────────
# 0. Config
# ─────────────────────────────────────────────
MODEL_ID        = "typhoon-ai/typhoon-7b"
OUTPUT_DIR      = "./typhoon-3.5b-init"
HF_REPO         = "Phonsiri/typhoon-3.5b-init"   # ← Hugging Face repo id
PRIVATE_REPO    = False                           # set True to make repo private
TEACHER_LAYERS  = 32
STUDENT_LAYERS  = 16          # keep even-indexed layers: 0,2,4,...,30
LOAD_DTYPE      = torch.bfloat16   # change to None for full fp32 on CPU


# ─────────────────────────────────────────────
# 1. Load Config & Tokenizer
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 1 │ Loading config & tokenizer …")
print("=" * 60)

config = AutoConfig.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"  Original num_hidden_layers : {config.num_hidden_layers}")
config.num_hidden_layers = STUDENT_LAYERS
print(f"  Updated  num_hidden_layers : {config.num_hidden_layers}")


# ─────────────────────────────────────────────
# 2. Initialize Models
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 2 │ Loading teacher model (this may take a while) …")
print("=" * 60)

load_kwargs = dict(
    device_map="cpu",       # keep everything on RAM to avoid VRAM pressure
    torch_dtype=LOAD_DTYPE,
    trust_remote_code=True,
)
teacher = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
teacher.eval()
print(f"  Teacher loaded  ({TEACHER_LAYERS} layers)")

print("\nStep 2 │ Creating empty student model …")
student = AutoModelForCausalLM.from_config(config)
student = student.to(dtype=LOAD_DTYPE)
student.eval()
print(f"  Student created ({STUDENT_LAYERS} layers)")


# ─────────────────────────────────────────────
# 3. Weight Copying
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 │ Copying weights from teacher → student …")
print("=" * 60)

with torch.no_grad():
    # 3-a  Shared / non-layer weights
    print("  Copying embed_tokens …")
    student.model.embed_tokens.load_state_dict(
        teacher.model.embed_tokens.state_dict()
    )

    print("  Copying final norm …")
    student.model.norm.load_state_dict(
        teacher.model.norm.state_dict()
    )

    print("  Copying lm_head …")
    student.lm_head.load_state_dict(
        teacher.lm_head.state_dict()
    )

    # 3-b  Transformer layers (Even-layer / Skip-layer pruning)
    #      student layer i  ←  teacher layer 2*i
    print(f"\n  Copying {STUDENT_LAYERS} transformer layers (even-skip) …")
    print(f"  {'Student Layer':<16} ← {'Teacher Layer'}")
    print(f"  {'-'*35}")
    for i in range(STUDENT_LAYERS):
        src_idx = i * 2          # 0,2,4,…,30
        student.model.layers[i].load_state_dict(
            teacher.model.layers[src_idx].state_dict()
        )
        print(f"  layers[{i:>2}]         ←  layers[{src_idx:>2}]")

print("\n  ✅ All weights copied successfully.")


# ─────────────────────────────────────────────
# 4. Save Pruned Model & Tokenizer
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 4 │ Saving pruned model to '{OUTPUT_DIR}' …")
print("=" * 60)

student.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"  ✅ Saved locally → {OUTPUT_DIR}")

# ─────────────────────────────────────────────
# 5. Push to Hugging Face Hub
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 5 │ Pushing to Hugging Face Hub: {HF_REPO} …")
print("=" * 60)

student.push_to_hub(
    HF_REPO,
    private=PRIVATE_REPO,
    commit_message="Add typhoon-3.5b-init (16-layer even-skip pruned from typhoon-7b)",
)
tokenizer.push_to_hub(
    HF_REPO,
    private=PRIVATE_REPO,
    commit_message="Add tokenizer for typhoon-3.5b-init",
)
print(f"  ✅ Model pushed → https://huggingface.co/{HF_REPO}")


# ─────────────────────────────────────────────
# 6. Summary
# ─────────────────────────────────────────────
total_params = sum(p.numel() for p in student.parameters())
trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)

print("\n" + "=" * 60)
print("✅ All Done!")
print("=" * 60)
print(f"  Source model   : {MODEL_ID}")
print(f"  Local dir      : {OUTPUT_DIR}")
print(f"  HF repo        : https://huggingface.co/{HF_REPO}")
print(f"  Teacher layers : {TEACHER_LAYERS}")
print(f"  Student layers : {STUDENT_LAYERS}")
print(f"  Total params   : {total_params / 1e9:.3f}B")
print(f"  Trainable      : {trainable_params / 1e9:.3f}B")
print(f"  dtype          : {LOAD_DTYPE}")
print("=" * 60)
print("Next step: use ./typhoon-3.5b-init (or HF repo) as base for CPT training.")
