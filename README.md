# The Weather Theme — Typhoon CPT Distillation Pipeline

Thai Continued Pre-Training (CPT) pipeline ด้วย Knowledge Distillation จาก **typhoon-ai/typhoon-7b** (32 layers) ลงสู่ **Student Model** ขนาด ~3.5B (16 layers) ตามหลักการในเปเปอร์ Typhoon (2312.13951)

---

## 📁 โครงสร้างไฟล์

| ไฟล์ | หน้าที่ |
|---|---|
| `prune_typhoon_3_5b.py` | หั่น typhoon-7b จาก 32 → 16 layers (Even-skip) แล้ว push ขึ้น HF Hub |
| `prepare_thai_data.py` | เตรียม Dataset ภาษาไทย (CulturaX + Wikipedia) พร้อม Heuristic Filter + Tokenize + Pack เป็น Block 4096 tokens แล้วดันขึ้น Hub ทันที |
| `cpt_distill_train.py` | เทรน Distillation CPT บน Cloud GPU — Auto-resume จาก HF Hub ทุก Session |

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 🚀 วิธีใช้งาน

### Step 1 — หั่นโมเดล (รันบนเครื่องตัวเองครั้งเดียว)

```bash
python prune_typhoon_3_5b.py
# → สร้าง ./typhoon-3.5b-init/ และ push ขึ้น Phonsiri/typhoon-3.5b-init
```

### Step 2 — เตรียม Dataset (รันบน Cloud ก่อนเทรน)

```bash
# ตั้ง HF Token ก่อน เพื่อให้ Push ได้
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
python prepare_thai_data.py
# → สร้าง ./thai_cpt_data/
# → Push อัตโนมัติขึ้นไปที่ Phonsiri/thai-cpt-3.5b-data
```

### Step 3 — เทรน (ทุกวัน วันละ 5 ชั่วโมง)

```bash
# ตั้ง HF Token (แทน huggingface-cli login)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx

# วันแรก
python cpt_distill_train.py --skip-prune

# วันต่อไป (auto-resume จาก checkpoint ล่าสุดบน HF Hub)
python cpt_distill_train.py --skip-prune
```

---

## 🧠 Architecture

- **Teacher**: `typhoon-ai/typhoon-7b` — 32 layers, frozen (ไม่ update gradient)
- **Student**: `Phonsiri/typhoon-3.5b-init` — 16 layers (เลเยอร์คู่ 0,2,4,...,30 ของ Teacher)
- **Projection**: Linear layer ปรับ hidden dim Student → Teacher (per layer)

### Loss Function

```
Loss = 0.3 × L_CE  +  0.5 × L_KD  +  0.2 × L_Hidden

L_CE     = Cross-Entropy (next-token prediction)
L_KD     = KL-Divergence (student logits vs teacher logits, T=2)
L_Hidden = MSE (student hidden states vs teacher hidden states)
```

---

## 🔄 Auto-Resume Flow

```
Session ใหม่ (เครื่อง Cloud ใหม่)
    ↓
Python login ด้วย HF_TOKEN
    ↓
ดึง checkpoint ล่าสุดจาก Phonsiri/typhoon-3.5b-cpt-ckpt
    ↓
โหลด model.pt + optimizer.pt + scheduler.pt + projectors.pt
    ↓
เทรนต่อจาก global_step ที่บันทึกไว้ใน meta.json
    ↓
Save ทุก 25 steps → Push ขึ้น HF Hub ทันที
```

---

## 📊 Dataset Filtering (Typhoon §3.1 Heuristics)

| Filter | Threshold |
|---|---|
| Thai char ratio | ≥ 40% |
| Special char ratio | ≤ 15% |
| Document length | 200 – 100,000 chars |
| Mean line length | 20 – 1,500 chars |
| Ellipsis line ratio | ≤ 30% |

Sources: **uonlp/CulturaX (th)** + **wikimedia/wikipedia (th)**  
Tokenizer: `typhoon-ai/typhoon-7b` (2.62× more efficient than GPT-4 for Thai)

---

## 🖥️ Hardware Target

- GPU: 1× NVIDIA H100 (80 GB VRAM)
- Precision: bfloat16 (SDPA, no FlashAttention-2)
- Batch size: 4 × grad_accum 8 = effective 32
- Context length: 4096 tokens
- Session limit: 5 hours/day

---

## 📦 HF Hub Repos

| Repository | หน้าที่ |
|---|---|
| `typhoon-ai/typhoon-7b` | Teacher Model (ต้นฉบับ) |
| `Phonsiri/typhoon-3.5b-init` | Student Model หลัง Pruning |
| `Phonsiri/typhoon-3.5b-cpt-ckpt` | Checkpoint Backup (auto-push ระหว่างเทรน) |
| `Phonsiri/thai-cpt-3.5b-data` | Pre-tokenized Dataset 4096-token blocks (ใช้ stream ตอนเทรน) |
