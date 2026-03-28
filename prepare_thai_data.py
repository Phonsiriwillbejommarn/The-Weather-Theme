"""
prepare_thai_data.py
══════════════════════════════════════════════════════════════════════════════
Thai CPT Data Preparation Pipeline  (based on Typhoon paper §3.1)

Steps:
  1. Stream cc100 (th) + uonlp/CulturaX (th) from HuggingFace
  2. Heuristic filtering  (char ratio, line length, doc length)
  3. Tokenize with typhoon-ai/typhoon-7b tokenizer
  4. Pack into 4096-token blocks and save as Arrow shards on disk

Output: 
  1. ./thai_cpt_data/  (Arrow dataset shards locally)
  2. Pushed to Hugging Face Hub as Phonsiri/thai-cpt-3.5b-data
Requirements:
    pip install transformers datasets tqdm
══════════════════════════════════════════════════════════════════════════════
"""

import re
import os
import unicodedata
from pathlib import Path
from typing import Iterator

from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 0.  Config
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_ID        = "typhoon-ai/typhoon-7b"
OUTPUT_DIR          = "./thai_cpt_data"
HF_DATASET_REPO     = "Phonsiri/thai-cpt-3.5b-data" # Where to upload the result
BLOCK_SIZE          = 4096          # Typhoon context length
SHARD_SIZE          = 50_000        # blocks per shard file
MAX_SHARDS          = None          # set int to limit (None = unlimited)
MAX_EXAMPLES_PER_SOURCE = 500_000   # raw docs per source (None = all)

# ── Source datasets ───────────────────────────────────────────────────────────
SOURCES = [
    # (dataset_id, subset/config, split)
    # CulturaX Thai — high-quality multilingual, deduplicated
    ("uonlp/CulturaX",      "th",           "train"),
    # Wikipedia Thai — clean, factual, public Parquet
    ("wikimedia/wikipedia", "20231101.th",  "train"),
]

# ── Heuristic Filter Thresholds  (Typhoon / Falcon RefinedWeb style) ─────────
MIN_THAI_RATIO          = 0.40    # ≥ 40% Thai Unicode characters
MAX_SPECIAL_CHAR_RATIO  = 0.15    # ≤ 15% special/punctuation characters
MIN_DOC_CHARS           = 200     # discard very short documents
MAX_DOC_CHARS           = 100_000 # discard extremely long (likely SEO spam)
MIN_MEAN_LINE_LEN       = 20      # avg chars per line; below this = bullet spam
MAX_MEAN_LINE_LEN       = 1500    # above this = wall-of-text, likely SEO
MAX_LINE_ELLIPSIS_RATIO = 0.30    # ≤ 30% lines ending with "…" / "..."
MIN_WORD_COUNT          = 20      # minimum whitespace-tokenised "words"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Heuristic Filters
# ─────────────────────────────────────────────────────────────────────────────
# Thai Unicode range: U+0E00 – U+0E7F
_THAI_RE    = re.compile(r"[\u0E00-\u0E7F]")
_SPECIAL_RE = re.compile(r"[^\w\s\u0E00-\u0E7F]", re.UNICODE)


def thai_char_ratio(text: str) -> float:
    total = len(text)
    return len(_THAI_RE.findall(text)) / total if total else 0.0


def special_char_ratio(text: str) -> float:
    total = len(text)
    return len(_SPECIAL_RE.findall(text)) / total if total else 0.0


def mean_line_length(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    return sum(len(l) for l in lines) / len(lines)


def ellipsis_line_ratio(text: str) -> float:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0
    ellipsis_lines = sum(1 for l in lines if l.rstrip().endswith(("…", "...")))
    return ellipsis_lines / len(lines)


def passes_heuristic_filter(text: str) -> bool:
    """Return True if the document passes all Typhoon-style heuristics."""
    n = len(text)
    if n < MIN_DOC_CHARS or n > MAX_DOC_CHARS:
        return False

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        return False

    if thai_char_ratio(text) < MIN_THAI_RATIO:
        return False

    if special_char_ratio(text) > MAX_SPECIAL_CHAR_RATIO:
        return False

    mll = mean_line_length(text)
    if mll < MIN_MEAN_LINE_LEN or mll > MAX_MEAN_LINE_LEN:
        return False

    if ellipsis_line_ratio(text) > MAX_LINE_ELLIPSIS_RATIO:
        return False

    return True


def normalize_text(text: str) -> str:
    """Light normalization: NFC + collapse repeated whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Token Block Packer
# ─────────────────────────────────────────────────────────────────────────────
def pack_into_blocks(
    token_stream: Iterator[list[int]],
    block_size: int,
    eos_id: int,
) -> Iterator[dict]:
    """
    Concatenate token lists and slice into fixed-length blocks.
    The last partial block is discarded (avoids padding).
    Each block is yielded as {"input_ids": [...], "labels": [...]}.
    Labels == input_ids shifted by 1 (causal LM).
    """
    buf: list[int] = []
    for ids in token_stream:
        buf.extend(ids)
        buf.append(eos_id)            # document boundary

        while len(buf) >= block_size + 1:
            chunk = buf[: block_size + 1]
            buf   = buf[block_size:]
            yield {
                "input_ids": chunk[:-1],
                "labels":    chunk[1:],
            }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-source streaming pipeline
# ─────────────────────────────────────────────────────────────────────────────
def stream_filtered_texts(dataset_id: str, subset: str, split: str) -> Iterator[str]:
    """Stream raw texts from a HuggingFace dataset, apply heuristic filter."""
    try:
        ds = load_dataset(
            dataset_id,
            subset,
            split=split,
            streaming=True,
        )
    except Exception as e:
        print(f"  [WARNING] Could not load {dataset_id}/{subset}: {e}")
        return

    # Detect text column
    first = next(iter(ds))
    text_col = "text" if "text" in first else list(first.keys())[0]

    passed = 0
    total  = 0
    for ex in ds:
        total += 1
        if MAX_EXAMPLES_PER_SOURCE and total > MAX_EXAMPLES_PER_SOURCE:
            print(f"    {dataset_id}/{subset}: reached MAX_EXAMPLES_PER_SOURCE={MAX_EXAMPLES_PER_SOURCE:,} — stopping.")
            break
        text = ex.get(text_col, "")
        if not isinstance(text, str) or not text:
            continue
        text = normalize_text(text)
        if passes_heuristic_filter(text):
            passed += 1
            yield text
        if total % 100_000 == 0:
            print(f"    {dataset_id}/{subset}: {passed}/{total} passed "
                  f"({100*passed/total:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Thai CPT Data Preparation  (Typhoon §3.1 heuristics)")
    print("=" * 70)

    # ── Load Tokenizer ────────────────────────────────────────────────────────
    print(f"\n[tokenizer] Loading {TOKENIZER_ID} …")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    eos_id = tokenizer.eos_token_id
    print(f"  vocab size : {tokenizer.vocab_size:,}")
    print(f"  EOS id     : {eos_id}")
    print(f"  block size : {BLOCK_SIZE} tokens")

    # ── Track overall stats ───────────────────────────────────────────────────
    shard_idx   = 0
    total_blocks = 0
    buffer_blocks: list[dict] = []

    def flush_shard():
        nonlocal shard_idx, total_blocks
        if not buffer_blocks:
            return
        shard_ds = Dataset.from_list(buffer_blocks)
        shard_path = out_dir / f"shard_{shard_idx:05d}"
        shard_ds.save_to_disk(str(shard_path))
        total_blocks += len(buffer_blocks)
        print(f"\n[shard {shard_idx:05d}] Saved {len(buffer_blocks):,} blocks "
              f"→ {shard_path}  (total so far: {total_blocks:,})")
        shard_idx += 1
        buffer_blocks.clear()

    # ── Iterate sources ───────────────────────────────────────────────────────
    for ds_id, subset, split in SOURCES:
        print(f"\n{'─'*70}")
        print(f"[source] {ds_id}  config={subset}  split={split}")
        print(f"{'─'*70}")

        def token_stream_for_source():
            for text in stream_filtered_texts(ds_id, subset, split):
                ids = tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    yield ids

        pbar_desc = f"{ds_id.split('/')[-1]}/{subset}"
        with tqdm(desc=pbar_desc, unit=" blocks") as pbar:
            for block in pack_into_blocks(token_stream_for_source(), BLOCK_SIZE, eos_id):
                buffer_blocks.append(block)
                pbar.update(1)

                if len(buffer_blocks) >= SHARD_SIZE:
                    flush_shard()
                    pbar.reset()

                    if MAX_SHARDS and shard_idx >= MAX_SHARDS:
                        print(f"[stop] MAX_SHARDS={MAX_SHARDS} reached.")
                        flush_shard()
                        break

        # Flush any remaining blocks from this source before moving on
        if MAX_SHARDS and shard_idx >= MAX_SHARDS:
            break

    # Final shard
    flush_shard()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_tokens = total_blocks * BLOCK_SIZE
    print("\n" + "=" * 70)
    print("✅ Data Preparation Complete!")
    print("=" * 70)
    print(f"  Output dir     : {out_dir.resolve()}")
    print(f"  Shards saved   : {shard_idx}")
    print(f"  Total blocks   : {total_blocks:,}")
    print(f"  Total tokens   : {total_tokens:,}  ({total_tokens/1e9:.2f}B)")
    print(f"  Block size     : {BLOCK_SIZE} tokens")
    print("=" * 70)

    # ── Push to Hugging Face Hub ─────────────────────────────────────────────
    print(f"\n[hub] Pushing prepared dataset to {HF_DATASET_REPO} …")
    try:
        from datasets import load_from_disk
        shard_dirs = sorted([d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("shard_")])
        if shard_dirs:
            shards = [load_from_disk(str(s)) for s in shard_dirs]
            full_ds = concatenate_datasets(shards)
            full_ds.push_to_hub(HF_DATASET_REPO, private=True)
            print(f"[hub] ✅ Successfully published to https://huggingface.co/datasets/{HF_DATASET_REPO}")
        else:
            print("[hub] ⚠️  No shards found to push.")
    except Exception as e:
        print(f"[hub] ❌ Failed to push to hub: {e}")
        print("[hub] Did you `export HF_TOKEN=...` before running?")


if __name__ == "__main__":
    main()
