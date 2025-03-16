# Author: Yuanjun "Dastin" Huang
# DDP pretraining for qwen2.5_0.5B on fineweb-edu
# Last updated on 03/15/2025 (MM/DD/YYYY)

from torchdata.stateful_dataloader.stateful_dataloader import StatefulDataLoader
import torch.distributed as dist
import torch.distributed
import torch.optim as optim
from datasets import load_dataset
import torch.utils
import torch.utils.checkpoint
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
# from LARC import LARC
import numpy as np
from torcheval.metrics.functional.text import perplexity
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch
import os
import shutil
import contextlib
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# from bitsandbytes.optim import LARS

DEBUG_STREAMING = False
HF_DATA_FIELD = "text"
HF_DATA_PATH = "HuggingFaceFW/fineweb-edu"
HF_DATA_SUBSET_NAME = "sample-10BT"
HF_DATA_SPLIT = "train"
MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
TOKENIZER_NAME = 'gpt2'
MAX_LR = 5e-4
MIN_LR = 5e-5
WARMUP_STEP = 2000
MAX_STEP = 100000


def lr_scheduler_fn(x):
    if x < WARMUP_STEP:
        return x / WARMUP_STEP
    else:
        if x > MAX_STEP:
            x = MAX_STEP
        return (MIN_LR + (MAX_LR-MIN_LR) * np.cos((x-WARMUP_STEP)/(MAX_STEP-WARMUP_STEP)*np.pi/2)) / MAX_LR


class CollateFn:
    def __init__(self, tokenizer, max_length=2048):
        if type(tokenizer) == str:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer, padding_side='left')
        else:
            self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        self.max_length = max_length

    def collate_fn(self, batch):
        text = [x['text'] for x in batch]
        x = self.tokenizer(
            text, return_tensors='pt',
            max_length=self.max_length,
            # padding='max_length',
            padding=True,
            truncation=True,
        )
        y = deepcopy(x['input_ids'])
        y[x['input_ids'] == self.tokenizer.pad_token_id] = -100
        return x, y


class Trainer:
    def __init__(self, save_frequency=1000, clip_grad_max_norm=1, save_dir=None, log_dir='', save_top_k=2, previous_ckpt_step=0, load_from_ckpt=None):
        self.save_frequency = save_frequency
        self.clip_grad_max_norm = clip_grad_max_norm
        self.save_top_k = save_top_k
        self.step_cnt = 0
        self.top_k_list = []
        self.previous_ckpt_step = previous_ckpt_step

        if save_dir is not None:
            save_dir = 'model_checkpoints/{}'.format(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("cpu:gloo,cuda:nccl")
        self.rank = dist.get_rank()
        if self.rank != 0:
            from datasets.utils.logging import disable_progress_bar
            disable_progress_bar()

        self.train_dataset = load_dataset(
            HF_DATA_PATH,
            name=HF_DATA_SUBSET_NAME,
            split=HF_DATA_SPLIT+'[:98%]',
            # split=HF_DATA_SPLIT,
            streaming=DEBUG_STREAMING,
        )
        self.val_dataset = load_dataset(
            HF_DATA_PATH,
            name=HF_DATA_SUBSET_NAME,
            split=HF_DATA_SPLIT+'[-2%:]',
            # split=HF_DATA_SPLIT+'[-5000:]',
            # split=HF_DATA_SPLIT,
            streaming=DEBUG_STREAMING,
        )

        if self.rank == 0:
            print('train: {}, val {}'.format(
                len(self.train_dataset), len(self.val_dataset)))
            log_dir = 'runs/{}'.format(log_dir)
            self.writer = SummaryWriter(log_dir=log_dir)

        self._init_model_and_optimizer(load_from_ckpt)

        self.lr_scheduler = LambdaLR(
            self.optimizer, lr_scheduler_fn)
        self.lr_scheduler.last_epoch = previous_ckpt_step

        self.model_state_ckpt_future = None
        self.optim_state_ckpt_future = None

    def _init_model_and_optimizer(self, load_from_ckpt):
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.vocab_size = 50258
        config.bos_token_id = 50256
        config.eos_token_id = 50256
        config.use_cache = False

        model = Qwen2ForCausalLM(config).to('cuda')
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        self.ddp_model = DDP(model, device_ids=[self.rank])
        self.optimizer = optim.AdamW(self.ddp_model.parameters(),
                                     lr=MAX_LR, weight_decay=0.1)

        if load_from_ckpt is not None and self.rank == 0:
            # torch.distributed.checkpoint.state_dict_loader.load(self.ddp_model.state_dict(
            # ), checkpoint_id=os.path.join(load_from_ckpt, 'model'))
            # torch.distributed.checkpoint.state_dict_loader.load(self.optimizer.state_dict(
            # ), checkpoint_id=os.path.join(load_from_ckpt, 'optim'))
            self.ddp_model.load_state_dict(torch.load(
                os.path.join(load_from_ckpt, 'model.pt'), weights_only=True))
            self.optimizer.load_state_dict(torch.load(
                os.path.join(load_from_ckpt, 'optim.pt'), weights_only=True))

        self.ddp_model.train()

    def save_checkpoint(self, path):
        if self.rank == 0:
            os.makedirs(path)
            try:
                torch.save(self.ddp_model.state_dict(),
                           os.path.join(path, 'model.pt'))
            except Exception as e:
                print('failed to save model state on rank {}\n{}'.format(
                    self.rank, e))

            try:
                torch.save(self.optimizer.state_dict(),
                           os.path.join(path, 'optim.pt'))
            except Exception as e:
                print('failed to save optimizer state on rank {}\n{}'.format(
                    self.rank, e))
        dist.barrier()

    def validate(self, val_dataloader):
        torch.cuda.empty_cache()
        self.ddp_model.eval()
        with torch.no_grad():
            val_loss = 0
            val_ppl = 0
            batch_cnt = torch.tensor(0).to('cuda')
            val_iter = iter(val_dataloader)
            if self.rank == 0:
                progress = tqdm(val_iter)
                print('calculating validation loss')
            else:
                progress = val_iter
            for data in progress:
                x, y = data
                pred = self.ddp_model(**x.to('cuda'), labels=y.to('cuda'))
                val_loss += pred['loss']
                val_ppl += perplexity(pred['logits']
                                      [:, :-1, :], y[:, 1:], ignore_index=-100)
                batch_cnt += 1

        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ppl, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_cnt, op=dist.ReduceOp.SUM)
        val_loss /= batch_cnt
        val_ppl /= batch_cnt
        self.ddp_model.train()
        torch.cuda.empty_cache()
        if self.rank == 0:
            print('Validation loss calculated: {}'.format(val_loss))
        return val_loss, val_ppl

    def train_on_dataloader(self, train_sampler, grad_accu_step):
        if self.rank == 0:
            pbar = tqdm(total=len(self.train_dataloader)//grad_accu_step)
        progress = enumerate(self.train_dataloader)

        for epoch in range(1):
            train_sampler.set_epoch(epoch)
            for i, data in progress:
                if self.step_cnt < self.previous_ckpt_step:
                    if i % grad_accu_step == grad_accu_step - 1:
                        self.step_cnt += 1
                        if self.rank == 0:
                            pbar.update(1)
                    continue
                x, y = data
                if i % grad_accu_step == 0:
                    mini_batch_loss = 0
                    mini_batch_ppl = 0
                    self.optimizer.zero_grad(set_to_none=True)

                if i % grad_accu_step == grad_accu_step - 1:
                    sync_context = contextlib.nullcontext()
                else:
                    sync_context = self.ddp_model.no_sync()
                with sync_context:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred = self.ddp_model(
                            **x.to('cuda'), labels=y.to('cuda'))

                loss = pred['loss'] / grad_accu_step
                loss.backward()
                mini_batch_loss += loss.detach()
                mini_batch_ppl += perplexity(pred['logits']
                                             [:, :-1, :], y[:, 1:], ignore_index=-100)

                if i % grad_accu_step == grad_accu_step - 1:
                    mini_batch_ppl /= grad_accu_step
                    param_norm = torch.nn.utils.get_total_norm(
                        self.ddp_model.parameters())
                    grad_norm_pre_optimization = torch.nn.utils.clip_grad_norm_(
                        self.ddp_model.parameters(), self.clip_grad_max_norm)

                    dist.reduce(mini_batch_loss, 0, op=dist.ReduceOp.AVG)
                    dist.reduce(mini_batch_ppl, 0, op=dist.ReduceOp.AVG)

                    self.lr_scheduler.step()
                    self.optimizer.step()
                    grad_norm_post_optimization = torch.nn.utils.get_total_norm(
                        [x.grad for x in self.ddp_model.parameters()])

                    self.step_cnt += 1
                    if self.rank == 0:
                        self.writer.add_scalars(
                            'grad_norm', {
                                'param': param_norm,
                                'pre_opt_step': grad_norm_pre_optimization,
                                'post_opt_step': grad_norm_post_optimization
                            },
                            self.step_cnt)
                        self.writer.add_scalars(
                            'loss', {'train': mini_batch_loss}, self.step_cnt)
                        self.writer.add_scalars(
                            'perplexity', {'train': mini_batch_ppl}, self.step_cnt)
                        lr = self.lr_scheduler.get_last_lr()[0]
                        self.writer.add_scalar(
                            'learning_rate', lr, self.step_cnt)
                        self.writer.add_scalar('grad_accu_step',
                                               grad_accu_step, self.step_cnt)
                        postfix = {'step_cnt': self.step_cnt,
                                   'loss': mini_batch_loss.item(),
                                   'perplexity': mini_batch_ppl.item()
                                   }

                        pbar.set_postfix(postfix)

                    if self.step_cnt % self.save_frequency == 0:
                        val_loss = 0
                        val_loss, val_ppl = self.validate(self.val_dataloader)

                        if self.save_dir is not None:
                            if len(self.top_k_list) == self.save_top_k and self.top_k_list[-1]['loss'] < val_loss:
                                # no need to save
                                pass
                            else:
                                save_path = os.path.join(
                                    self.save_dir, '{}_{}_{:.2f}'.format(epoch, self.step_cnt, val_loss))
                                if self.rank == 0:
                                    if len(self.top_k_list) > 0:
                                        ckpt_path_to_remove = self.top_k_list[-1]['path']
                                        if len(self.top_k_list) == self.save_top_k:
                                            if os.path.isfile(ckpt_path_to_remove):
                                                os.remove(ckpt_path_to_remove)
                                            elif os.path.isdir(ckpt_path_to_remove):
                                                shutil.rmtree(
                                                    ckpt_path_to_remove)
                                            else:
                                                pass
                                            self.top_k_list.pop(-1)

                                    self.top_k_list.append({
                                        'path': save_path,
                                        'loss': loss,
                                    })
                                    self.top_k_list = sorted(
                                        self.top_k_list, key=lambda x: x['loss'])
                                self.save_checkpoint(save_path)
                                if self.rank == 0:
                                    print('model saved to {}'.format(save_path))

                        if self.rank == 0:
                            self.writer.add_scalars(
                                'perplexity', {'val': val_ppl}, self.step_cnt)
                            self.writer.add_scalars(
                                'loss', {'val': val_loss}, self.step_cnt)

                    if self.rank == 0:
                        pbar.update(1)

    def train(self):
        max_len_steps = [0, 256, 512, 768, 1024, 1536, 2048, 8192]
        batch_size_steps = [56, 28, 19, 14, 9, 7, 1]
        grad_acc_steps = [2, 4, 6, 8, 12, 16, 112]
        collate_fn_generator = CollateFn(TOKENIZER_NAME)

        for low, high, batch_size, grad_acc_step in zip(max_len_steps[:-1], max_len_steps[1:], batch_size_steps, grad_acc_steps):
            if self.rank == 0:
                print('max length: {}, batch size: {}, gradient accumulation steps: {}'.format(
                    high, batch_size, grad_acc_step))

            current_train_dataset = self.train_dataset.filter(
                lambda x: low < x['token_count'] < high
            )

            current_val_dataset = self.val_dataset.filter(
                lambda x: low < x['token_count'] < high
            )

            collate_fn_generator.max_length = high
            collate_fn = collate_fn_generator.collate_fn
            train_sampler = DistributedSampler(current_train_dataset)
            self.train_dataloader = StatefulDataLoader(current_train_dataset, shuffle=False,
                                                       sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, num_workers=4)
            val_sampler = DistributedSampler(current_val_dataset)
            self.val_dataloader = StatefulDataLoader(current_val_dataset, shuffle=False,
                                                     sampler=val_sampler, collate_fn=collate_fn, batch_size=batch_size, num_workers=2)
            self.train_on_dataloader(
                train_sampler, grad_acc_step)

        dist.destroy_process_group()


def main(**kwargs):
    trainer = Trainer(**kwargs)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_frequency', default=1000, type=int)
    parser.add_argument('--save_top_k', default=5, type=int)
    parser.add_argument('--clip_grad_max_norm', default=1, type=float)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--log_dir', default='test', type=str)
    parser.add_argument('--seed', default=163, type=int)
    parser.add_argument('--previous_ckpt_step', default=0, type=int)
    parser.add_argument('--load_from_ckpt', default=None, type=str)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(
        save_frequency=args.save_frequency,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_top_k=args.save_top_k,
        clip_grad_max_norm=args.clip_grad_max_norm,
        previous_ckpt_step=args.previous_ckpt_step,
        load_from_ckpt=args.load_from_ckpt,
    )
