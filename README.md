# LM-from-scratch

## What is this about?

An end2end recipe for LLM training containing pretraining, SFT and alignment. (Perhaps too big a dream? I'll try my best:))

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