# LM Training From Scratch

## What's this?

An end2end recipe covering language model pretrain and post-train steps that can be run on (relatively) lower-end GPUs (think about some NVIDIA models with mid-20 GB VRAM). 

## Motivation

Almost every research engineer/scientist in the field of modern ML systems knows how mainstream auto-regressive LMs are trained: a next-token prediction pretrain on a large corpus followed by post-train steps involving SFT and RLHF. However, despite the simplicity of high-level idea, few people in this field have *actually* written code to reproduce the process.

Perhaps you'd ask, "why do you need to rebuild the wheel when there is already a pretty solid ecosystem including huggingface, trl and other libraries?" Indeed, most common tasks such as finetuning a pretrained LLM using a few hundread text corpus can be easily done by calling a few well-packaged APIs. However, things start to get spicy when you work on a "less common" task or a new algorithm that these frameworks don't yet support. Even a "mix and match" of very common tasks can be difficult.

Think about combining PEFT and FSDP to finetune a quantized pretrained LLM in a "high level API only" fashion. You'll be struck by incompatability in naming convention within the checkpoint, which raises error using peft's API to load the adaptor whose layer namings follow fsdp's nested pattern *(in fact, this topic alone could lead to a separate project and even result in committing a patch to `peft` library. I won't dig deep here since this repo is dedicated to LLM training pipeline)*. 

In a nutshell, there are many cases where writing custom code instead of calling well-established APIs is a necessity, especially for people working on cutting-edge AIML topics. This is when someone with solid understanding and *actual* implementation experience on LM training delivers while others struggle.

This project was inspired by a similar personal project [MyLLM](https://github.com/LF-Luis/MyLLM) by Luis Fernandez. I made the decision to work on an improved version when listening to Luis' presentation in SF last December. The goals are to sharpen my coding skill and deepen my understanding on how modern distributed training work. More specifically, I have a few action items:

* To use FSDP instead of DDP, since FSDP is the only option when training *large* model that don't fit within a single GPU
* To upscale the model size and enlarge context window while on a much tighter GPU buget. The original *MyLLM* project was powered by 8xA100(80GB). What if I can do this on 4 lower-end GPUs with 20-ish GB VRAM each?
* To implement post-train process

## Tech Details

I will not rely on any modern training libraries except for native `pytorch` and `transformers`. The only purpose of `transformers` is to import the model architecture; no training-related API from `transformer` is used. 

The 0.5B variant of Qwen2 architecture is selected with a shrinked embedding layer that only support 50k tokens. A stock Qwen model comes with a vocabulary with ~150k tokens, but a large portion of those are dedicated to non-english languages. Since this project focuses on *training* instead of the *model* itself, I decide to use a smaller gpt2 vocabulary to save GPU memory.

To enable training with longer context window on GPUs with very limited memory, gradient checkpointing, which is well-supported by `pytorch`, is turned on in every setup.

To efficiently utilize GPU memory, I'm using different batch sizes on corpus with different token counts. The training corpus is divided into several chunks, each with a different number of token limit, i.e., each sample in chunk 1's token count is within (0, 256], each sample in chunk 1's token count is within (256, 512], etc. ... The batch size of each chunk is inversly proportional to the max length of each chunk. This is because modern attention implementations (e.g., flash attention) have already lowered the *memory* complexity to *O(n)* where *n* is the tonken length, though the *arithmetic* complexity remains *O(n<sup>2</sup>)*. The chunk-specific batch size is set in such a way that maximum per-batch GPU memory consumption stays almost at the same leval across different chunks. This design is *not* supported if you use `trl.SFTTrainer` out of the box because the batch size must remain unchanged within a single `SFTTrainer.train()` call. This is yet another example showcasing the limits of existing frameworks.

### Pretrain

I regard this project as a coding exercise and treat it as a real-world areana to compare DDP against FSDP. `pytorch` supports 2 versions of FSDP natively. FSDP1 shards the entire model across multiple devices, and each parameter is sent to a specific device *as a whole*. FSDP2, however, wraps each single parameter (tensor) as an `DTensor` instance, and distributes different "chunk" of the single `DTensor` instance across multiple devices. If you make the analogy that FSDP1 shards the model "vertically", then FSDP2 does that "horizontally". Besides, FSDP2 also has a (slightly) different API compared to its predecessor.

Despite the difference between FSDP1 and FSDP2, they share the commonality that each rank stores a *different* partition of the entire state (entire state = model weights + optimizer state). On the contrary, under DDP each rank keeps an *identical* copy of the same state. For FSDP family, there are 2 ways to save the state to checkpoint: one saves each rank's state as a separate checkpoint, and the other gathers all partitions before merging them into a single checkpoint. I choose the first approach (a.k.a. distributed checkpointing) for its efficiency.

What's confusing is that there are also 2 different ways to save distributed checkpoints. Both are referred to in different sample codes from PyTorch's official documentation without a clear distinction. The first approach saves model weight and optimizer state separately with each divided into `WORLD_SIZE` partitions. The second approach wraps model weight and optimizer state into a single `torch.distributed.checkpoint.stateful.Stateful` instance, and the instance's state is split into `WORLD_SIZE` partitions. *In theory* either approaches should be compatible with both FSDP1 & FSDP2, creating 4 different "mix and match" solutions *(disclaimer: In practice I have only verified 3 out of these 4 choices)*.

#### DDP

`ddp_pretrain.py` implements a vanilla data parallelism training pipeline using basic `pytorch` DDP APIs. DDP has been integrated into `pytorch` for a long time and there's an abundance of documentation & sample codes online.

#### FSDP1

`fsdp1_pretrain.py` is built on top of `ddp_pretrain.py` with minor modifications on FSDP model wrapping, checkpoint saving/loading and inter-GPU synchronization. Like DDP, FSDP1 is also time-proven and well-documented.

#### FSDP2

Things start to get a bit spicy once we enter the realm of FSDP2. Theoretically FSDP2 should give you around 5% performance gain over FSDP1. Thus the official PyTorch documentation recommend that you should go with FSDP2 if you're building something new and don't need backward compatibility with FSDP1. However, there is very limited amount of documentation/example code online (including offical PyTorch website and 3rd party forums) when I did the experiment back in the first quarter of 2025. The good news is that PyTorch is on its way to deprecate FSDP1 and I've found this recently-updated [tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html). It's worth checking out whether you are new to FSDP world or are migrating from FSDP1 to FSDP2.

In `fsdp2_pretrain.py` I'm checkpointing the model weights and optimizer states separately instead of wrapping both in a single `torch.distributed.checkpoint.stateful.Stateful` instance as I did in `fsdp1_pretrain.py`. The level of control granularity and the complexity of code you prefer are the top 2 factors to consider when deciding which checkpointing approach to adopt.

### Post-Train

#### Supervised Finetuning

SFT is usually the first step in post training in most modern LLM training pipeline. Pretraining only gives you a good "next token predictor" that may not excel at instruction following. Conversation-like corpus consisting of "system", "user" and "assistant" components are usually adopted to showcase the LLM how to respond to a user's request. I use [`open-thoughts/OpenThoughts-114k`](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k), an open reasoning dataset curated with the help of advanced reasoning models, to finetune the pretrained model. 

The unique conversation corpus which distinguishes reasoning tokens from the "normal" output tokens introduces a few special tokens, e.g., `<|begin_of_thought|>`, `<|end_of_thought|>`, to name a few. Such special tokens aren't present in the vocabulary of GPT2, which is simply a vanilla auto-regressive next token predictor. In `fsdp2_sft.py` you can find codes dedicated to handle these extra tokens as well as a separate collate function to format conversation corpus into a single text string. Gradients contributed to the loss are only accumulated on tokens belonging to "assistant" components.

### Next Step?

I may continue to work on RL-based post-training algorithm in the future.

## Distributed Training Notes

We no longer live in an era when a single accelerator is enough for most tasks and distributed training/finetuning is almost inevitable for most ML practitioners. Below are some of my takeaways:

1. Familiarize yourself with various [collective operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html) first.
2. Process hangs are intimidating because oftentimes the error message *seems* to indicate a low level CUDA-specific issue. I found the best practice to examine the control flow of each rank. For example, in FSDP each rank needs to participate in checkpointing. When I was modifying `ddp_train.py` to use FSDP and forgot to uncomment `if self.rank == 0` from the `save_checkpoint()` method, rank 0 would keep waiting for the other ranks to save their share of the state, which never happened. In the end I was hit by process hang with a communication timeout error, which *seems* to be a CUDA library bug while in fact not.