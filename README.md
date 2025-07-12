# LM-from-scratch

An end-to-end PyTorch pipeline for training GPT-style transformers—from data preparation and tokenization through distributed pretraining (DDP, FSDP1/FSDP2) to supervised fine-tuning—with minimal dependencies. Optimized for GPUs with 20–30 GB VRAM.


## Overview

**LM-from-scratch** offers a hands-on, minimal-dependency training workflow using only PyTorch and Transformers. Ideal for developers who want full control and transparency, it runs efficiently on modest multi-GPU setups and is ready for scaling up.


## Features

- 🎯 **Multiple Training Modes**  
  Choose from DDP, FSDP v1, FSDP v2, or supervised fine-tuning (SFT) with zero high-level wrappers.

- 🔄 **Adaptive Batching by Token Count**  
  Employs bucketed batching that balances memory usage across context lengths.

- 💾 **Flexible Checkpointing**  
  Implementation of 2 different distributed checkpointing strategies

- 🧰 **Custom Collation & Masked Loss**  
  Special-token-aware collator for instruction-tuning corpora with masked loss on assistant tokens.


## Quick Start

```bash
git clone https://github.com/dastin359/LM-from-scratch.git
cd LM-from-scratch
pip install -r requirements.txt

# Pretrain using DDP on 4 GPUs
torchrun --nproc_per_node=4 ddp_pretrain.py \
  --save_dir=my_run --log_dir=my_log

# Pretrain using FSDP1 on 4 GPUs
torchrun --nproc_per_node=4 fsdp1_pretrain.py \
  --save_dir=my_run --log_dir=my_log

# Pretrain using FSDP2 on 4 GPUs
torchrun --nproc_per_node=4 fsdp2_pretrain.py \
  --save_dir=my_run --log_dir=my_log

# Perform supervised fine-tuning
torchrun --nproc_per_node=4 fsdp2_sft.py \
  --save_dir=sft_run --log_dir=sft_log
```

## Read More

For complete details—including project motivation, FSDP internals, token bucketing strategies, checkpoint formats, and distributed training insights—see [Technical Deep Dive](technical-deep-dive.md).