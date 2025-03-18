# LM-from-scratch

## What is this about?

An end2end recipe for language model training containing pretraining, SFT and alignment. (Perhaps too big a dream? I'll try my best:))

## Motivation

To be added!

## Tech Details

### Pretraining

#### FSDP2

Built upon PyTorch FSDP2 (equivalent to single node Zero stage 3)

Theoretically FSDP2 should give you around 5% performance gain over FSDP1. However there is very limited amount of documentation/example code online (compared to the quantity of resource for FSDP1). This is probably due to the fact that FSDP2 is still under active development and improvement (yes, I did come across a ton of issues along my journey) so that PyTorch team doesn't want to advertise it as a mature API.

The code shared here has been verified to run without bug on a single node 4 GPU setup.

#### FSDP

Built upon PyTorch FSDP (equivalent to single node Zero stage 3)

Code to be released soon.

#### DDP

Built with PyTorch DDP (vanilla data parallelism)

## Distributed Training Notes

Due to the sheer amount of compute needed to train a "modern" LM, ML practitioners often resort to distributed training instead of utilizing a single hardware accelerator (i.e., GPU). It adds an additional layer of complexity on top of single accelerator training due to the need to consider inter-node or intra-node communication and synchronization. 

`torch.distributed.barrier` is a useful tool providing an elegant handling of inter-accelerator synchronization. The rest of the notes focuses on the use of this tool in 3 different flavors of distributed training (DDP, FSDP1, FSDP2). It's vital to know how to use the tool in a correct maner in different setups.

Below I'll briefly share some of the hard lessons I learned.

### DDP

Since each accelerator maintains an exact copy of model weight & optimization state, we only need to pull data from a single accelerator during checkpointing. In the case when it takes some time to write the checkpoint to hard drive, it's intuitive to add a `barrier` towards the end of the `save_checkpoint` method, so as to bring all processes "on the same pace" again. That way, assuming checkpointing happens on rank 0, rank [1:N] essentially pause to wait for rank 0's storage writing.

However, in practice this actually leads to time out on one of the `all_reduce` operation. TODO: elaborate the cause.

### FSDP2

It's vital to place a `barrier` on each rank before checkpointing. Otherwise, chances are that you're trying to write some parameters not yet updated by the optimizer to your checkpoint because of a "slow" optimizer update on certain rank(s). To make things more confusing, program hang up due to time out on some collective operations may happen, which could lead the developer towards a wrong debugging direction.

TODO: elaborate why collective operation time out could happen.