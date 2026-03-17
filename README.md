# YORO - You Only Reason Once (Stage 2: Pretraining)

This repository contains the **Stage 2 pretraining** code for YORO, a novel LLM architecture that runs the expensive reasoning block exactly once per sequence instead of on every generated token.

The fine-tuning attempt is in [yoro-finetuning](https://github.com/SergiuDeveloper/yoro-finetuning). This repo takes a different approach: training the full YORO architecture from scratch on a large text corpus, without a teacher model.

---

## Motivation

Standard autoregressive LLMs repeat the full forward pass including every "reasoning" transformer layer - for every single generated token. Most of the per-token cost comes from these middle layers, not from selecting the next token.

**YORO's core idea:** run the heavy reasoning block exactly once (on the prompt), cache its output, and reuse it for all subsequent tokens. Small trainable subnets compensate for the missing reasoning passes so the model can still generate coherent continuations.

At inference time, for a sequence of T generated tokens, the "latent reasoning" cost is **O(1)** rather than **O(T)**.

---

## Architecture

YORO wraps a pretrained base model (here: `Qwen/Qwen2.5-0.5B`) and splits its transformer layers into three blocks, plus three small trainable subnets learned from scratch.

### Base model blocks

| Block                | Contents                                             | Role                                                |
| -------------------- | ---------------------------------------------------- | --------------------------------------------------- |
| **Embedding subnet** | First N transformer layers + token embeddings + RoPE | Converts tokens into contextualized representations |
| **Reasoning subnet** | Middle M transformer layers                          | Deep reasoning - run only once, on the prompt       |
| **Coherence subnet** | Final K transformer layers + LayerNorm + LM head     | Converts hidden states to output logits             |

In this pretraining run: N = K = 4 layers each, M = 16 layers (out of 24 total in Qwen2.5-0.5B).

### Trainable subnets

| Subnet                   | Type                                                                | Role                                                                                             |
| ------------------------ | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Adaptation subnet**    | MLP (Linear → ReLU, 4 layers)                                       | Transforms the cached reasoning output so it can be reused at later sequence positions           |
| **Compensation subnet**  | 4 freshly-initialized transformer layers (same class as base model) | Processes the current embedding-level representation to compensate for the absent reasoning pass |
| **Concatenation subnet** | MLP (Linear → ReLU, 4 layers)                                       | Merges adaptation and compensation outputs before the coherence block                            |

During pretraining, **all parameters are trained end-to-end** (both the base model blocks and the new subnets). The `freeze_base_model=False` flag enables this. The freeze logic is preserved in code for Stage 3 fine-tuning.

---

## Two forward modes

### Autoregressive mode (inference)

The model generates one token at a time, maintaining a reasoning cache across steps.

```
First token:
  tokens → embed_tokens → embedding_subnet → reasoning_subnet → [cache] → coherence_subnet → logits

All subsequent tokens:
  tokens → embed_tokens → embedding_subnet ─────────────────────────────┐
                                                                         ↓
                                              compensation_subnet ──→ concatenation_subnet → coherence_subnet → logits
                                              adaptation_subnet([cache]) ┘
```

The reasoning cache is computed once and never updated again.

### Teacher forcing mode (training)

To train efficiently, the full sequence (prompt + continuation) is processed in a single parallel forward pass. This is the key technical contribution that makes training tractable - without it, training would require running the autoregressive loop token-by-token, which is orders of magnitude slower.

The forward pass works as follows:

1. Embed the full sequence and run the **embedding subnet** over all positions.
2. Run the **reasoning subnet** only on the prompt portion `[0, prompt_length)` and cache the output.
3. Compute logits for prompt positions via the **coherence subnet** applied to the cached reasoning.
4. Pad the cached reasoning tensor to the full sequence length (zero-padding beyond `prompt_length`).
5. Run the **adaptation subnet** on the padded cache and the **compensation subnet** on the full embedding output - both in parallel over all positions.
6. Sum and pass through the **concatenation subnet**, then the **coherence subnet**, to get logits for generated positions `[prompt_length, seq_length)`.
7. Concatenate prompt and generated logits and compute cross-entropy loss against the shifted targets.

This masking structure faithfully simulates the autoregressive inference path (reasoning runs once, adaptation+compensation cover the rest) while allowing the entire sequence to be trained in a single batched forward pass.

---

## Training stages

### Stage 1 - Knowledge distillation fine-tuning ([yoro-finetuning](https://github.com/SergiuDeveloper/yoro-finetuning))

The first attempt at training YORO avoided pretraining from scratch. It started from a strong pretrained model (TinyLlama-1.1B-Chat), froze most of it, and trained only the three small subnets (adaptation, compensation, concatenation) to compensate for the missing reasoning passes.

A distillation setup was used because naive next-token prediction is insufficient when the student never sees ground-truth reasoning states. The teacher (original TinyLlama running normally) provided soft-label distributions (top-10 logprobs, temperature-scaled) rather than hard token labels. The teacher forcing masking mechanism allowed the full prompt+response sequence to be processed in a single parallel pass during training, making the approach tractable.

### Stage 2 - Full pretraining (this repo)

Stage 2 abandons fine-tuning in favor of training the full YORO architecture from scratch on a large corpus. Rather than distilling from a teacher, the model is pretrained end-to-end with standard cross-entropy loss on raw text.

All parameters are trained jointly - both the base model blocks (embedding, reasoning, coherence subnets initialized from Qwen2.5-0.5B weights) and the new trainable subnets (adaptation, compensation, concatenation, initialized randomly). The goal is to determine whether a model trained entirely through the YORO forward path can learn to produce high-quality outputs, and whether this yields a more efficient architecture class at scale.

---

## Pretraining run

| Parameter             | Value                                                                                |
| --------------------- | ------------------------------------------------------------------------------------ |
| Base model            | `Qwen/Qwen2.5-0.5B`                                                                  |
| Dataset               | [FineWeb-Edu sample-10BT](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) |
| Tokens trained        | 10 billion                                                                           |
| Context length        | 512 tokens                                                                           |
| Batch size            | 16 per GPU                                                                           |
| Gradient accumulation | 2 steps                                                                              |
| Effective batch size  | 256 sequences (16 × 2 accum × 8 GPUs)                                                |
| Learning rate         | 3e-4 (AdamW)                                                                         |
| Weight decay          | 0.01                                                                                 |
| Precision             | bfloat16                                                                             |
| Hardware              | 8 × H100 80GB                                                                        |
| Training time         | ~6 hours                                                                             |
| Framework             | [DeepSpeed](https://github.com/microsoft/DeepSpeed) ZeRO Stage 1                     |

All model weights are reinitialised from scratch (`REINITIALIZE_WEIGHTS = True`) - this is a **pretraining** run, not fine-tuning. The Qwen2.5-0.5B architecture is used as a structural template, not as a source of pretrained weights.

---

## Distributed training

Training is launched via [Modal](https://modal.com) using DeepSpeed's distributed data parallel backend.

- Each GPU processes an independent shard of the dataset (`split_dataset_by_node` from the HuggingFace `datasets` library).
- Gradients are synchronized across all 8 GPUs after each micro-batch via NCCL allreduce.
- **ZeRO Stage 1** shards optimizer states across GPUs, reducing per-GPU memory by ~4× for optimizer state without affecting the forward/backward pass.
- Flash Attention 2 is used for all transformer layers.
- The dataset (FineWeb-Edu, ~38 GB of parquet files) is pre-downloaded to a persistent Modal Volume to avoid network timeouts during training.

To reproduce:

```bash
# One-time: download dataset to Modal Volume
modal run pretrain_modal.py::prepare_data

# Launch training
modal run pretrain_modal.py
```

---

## Inference

**From a trained checkpoint:**

```bash
# First consolidate the DeepSpeed checkpoint (runs on Modal)
modal run consolidate_checkpoint.py --checkpoint-tag step_150000 --output best_model.pt

# Download from Modal Volume
modal volume get yoro-checkpoints best_model.pt best_model.pt

# Run inference locally
python infer_model.py
```

**From the raw Qwen2.5-0.5B baseline (for comparison):**

```bash
python infer_base.py
```

---

## Source

`subnet_model.py` defines all model classes:

- **`SubnetLLM`** - main model. Constructor arguments: `model_name`, `cache_dir`, `embedding_layers`, `coherence_layers`, `compensation_layers`, `adaptation_layers`, `concatenation_layers`, `device`, `dtype`, `freeze_base_model`.
- **`TransformerSubnet`** - wraps a slice of transformer layers (used for embedding, reasoning subnets).
- **`CompensationSubnet`** - same interface but layers are freshly initialized with random weights.
- **`MLPSubnet`** - stacked Linear → ReLU layers with Xavier-normal init (adaptation and concatenation subnets).
- **`CoherenceSubnet`** - wraps the final transformer layers, layer norm, and LM head.
