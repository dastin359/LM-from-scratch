# Author: Dastin (Yuanjun) Huang
# FSDP2 SFT for qwen2.5_0.5B on open-thoughts/OpenThoughts-114k
# Last updated on 07/06/2025 (MM/DD/YYYY)

from torchdata.stateful_dataloader.stateful_dataloader import StatefulDataLoader
import torch.distributed.checkpoint as dcp
import torch.distributed as dist
import torch.optim as optim
from datasets import load_dataset, load_from_disk
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, AutoConfig
from copy import deepcopy
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torcheval.metrics.functional.text import perplexity
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
)
import argparse
import torch
import os
import shutil
import math
import datetime
from app_state import AppState
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


DEBUG_STREAMING = False
HF_DATA_FIELD = "text"
# HF_DATA_PATH = 'open-thoughts/OpenThoughts-114k'
# You need to download 'open-thoughts/OpenThoughts-114k' and split it into train and val sets
# You can also use your own dataset, just make sure it has the same structure
HF_TRAIN_DATA_PATH = 'dataset/sft_train'
HF_VAL_DATA_PATH = 'dataset/sft_val'

MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
TOKENIZER_NAME = 'gpt2'
MAX_LR = 5e-4
MIN_LR = 5e-5
WARMUP_STEP = 50
MAX_STEP = 300


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
        self.tokenizer.add_tokens(
            [
                '<|begin_of_thought|>',
                '<|end_of_thought|>',
                '<|begin_of_solution|>',
                '<|end_of_solution|>',
                '<|im_start|>',
                '<|im_end|>',
            ]
        )
        self.max_length = max_length
        self.chat_template_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, padding_side='left')
        self.mask_template = torch.tensor([50262, 562, 10167])

    def _apply_chat_template(self, x):
        conversations = x['conversations']
        messages = [
            {"role": "system", "content": x['system']},
        ] + conversations
        tokenized_messages = self.chat_template_tokenizer.apply_chat_template(
            messages, tokenize=False)
        return tokenized_messages

    def collate_fn(self, batch):
        text = [self._apply_chat_template(x) for x in batch]
        x = self.tokenizer(
            text, return_tensors='pt',
            max_length=self.max_length,
            # padding='max_length',
            padding=True,
            truncation=True,
        )
        y = deepcopy(x['input_ids'])
        for idx in range(len(x['input_ids'])):
            msg = x['input_ids'][idx]
            for i in range(len(msg)-len(self.mask_template)):
                if torch.equal(msg[i:i+3], self.mask_template):
                    y[idx, :i] = -100
                    break
        return x, y


def is_sharded_fsdp_ckpt(path):
    try:
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(path, '.metadata'))
        for i in range(int(os.environ["WORLD_SIZE"])):
            assert os.path.exists(os.path.join(
                path, '__{}_0.distcp'.format(str(i))))
        return True
    except:
        return False


class Trainer:
    def __init__(self, save_frequency=1000, clip_grad_max_norm=1, save_dir=None, log_dir='', save_top_k=2, load_from_ckpt=None):
        self.save_frequency = save_frequency
        self.clip_grad_max_norm = clip_grad_max_norm
        self.save_top_k = save_top_k
        self.step_cnt = 0
        self.top_k_list = []

        if save_dir is not None:
            save_dir = 'model_checkpoints/{}'.format(save_dir)
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(
            # "cpu:gloo,cuda:nccl",
            "nccl",
            timeout=datetime.timedelta(seconds=300)
        )
        self.rank = dist.get_rank()
        if self.rank != 0:
            from datasets.utils.logging import disable_progress_bar
            disable_progress_bar()

        self.train_dataset = load_from_disk(
            HF_TRAIN_DATA_PATH,
        )
        self.val_dataset = load_from_disk(
            HF_VAL_DATA_PATH,
        )

        if self.rank == 0:
            print('train: {}, val {}'.format(
                len(self.train_dataset), len(self.val_dataset)))
            log_dir = 'runs/{}'.format(log_dir)
            self.writer = SummaryWriter(log_dir=log_dir)

        self._init_model_and_optimizer()

        if load_from_ckpt is not None:
            if is_sharded_fsdp_ckpt(load_from_ckpt):
                state_dict = {"app": AppState(self.fsdp_model, self.optimizer)}
                dcp.load(
                    state_dict=state_dict,
                    checkpoint_id=load_from_ckpt,
                )
                self.previous_ckpt_step = int(
                    load_from_ckpt.split('_')[-2])
            else:
                # find checkpoint
                loaded_ckpt_flag = False
                dirs = os.listdir(load_from_ckpt)
                steps = [int(x.split('_')[-2]) for x in dirs]
                dirs_with_steps = list(zip(dirs, steps))
                dirs_with_steps = sorted(
                    dirs_with_steps, reverse=True, key=lambda x: x[1])
                for dir, step in dirs_with_steps:
                    full_dir = os.path.join(load_from_ckpt, dir)
                    if is_sharded_fsdp_ckpt(full_dir):
                        try:
                            state_dict = {"app": AppState(
                                self.fsdp_model, self.optimizer)}
                            dcp.load(
                                state_dict=state_dict,
                                checkpoint_id=full_dir,
                            )

                            loaded_ckpt_flag = True
                            if self.rank == 0:
                                print('loaded checkpoint from {}'.format(full_dir))
                            self.previous_ckpt_step = step
                            break
                        except:
                            continue
                if not loaded_ckpt_flag:
                    self.previous_ckpt_step = 0
        else:
            self.previous_ckpt_step = 0

        dist.barrier()

        self.lr_scheduler = LambdaLR(
            self.optimizer, lr_scheduler_fn)
        self.lr_scheduler.last_epoch = self.previous_ckpt_step

        self.model_state_ckpt_future = None
        self.optim_state_ckpt_future = None

    def _init_model_and_optimizer(self):
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.vocab_size = 50304
        config.bos_token_id = 50256
        config.eos_token_id = 50256
        config.use_cache = False

        model = Qwen2ForCausalLM(config)
        ckpt = torch.load('model_checkpoints/qwen_fsdp1.pth',
                          weights_only=False)
        # the vocabulary size of pretrained model is 50256
        previous_emb_weight = deepcopy(
            ckpt['app']['model']['model.embed_tokens.weight'])
        ckpt['app']['model']['model.embed_tokens.weight'] = torch.zeros([
                                                                        50304, 896])
        ckpt['app']['model']['model.embed_tokens.weight'][:50258] = previous_emb_weight
        ckpt['app']['model']['lm_head.weight'] = ckpt['app']['model']['model.embed_tokens.weight']
        model.load_state_dict(ckpt['app']['model'])
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        self.fsdp_model = fully_shard(
            model,
            reshard_after_forward=True,
            mp_policy=mp_policy,
        )

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=MAX_LR, weight_decay=0.1)
        self.fsdp_model.train()

    def save_checkpoint(self, path):
        dist.barrier()
        state_dict = {"app": AppState(self.fsdp_model, self.optimizer)}
        dcp.save(state_dict, checkpoint_id=path)

    def validate(self, val_dataloader):
        torch.cuda.empty_cache()
        self.fsdp_model.eval()
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
                pred = self.fsdp_model(**x.to('cuda'), labels=y.to('cuda'))
                val_loss += pred['loss']
                val_ppl += perplexity(pred['logits']
                                      [:, :-1, :], y[:, 1:], ignore_index=-100)
                batch_cnt += 1

        dist.barrier()
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_ppl, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_cnt, op=dist.ReduceOp.SUM)
        val_loss /= batch_cnt
        val_ppl /= batch_cnt
        self.fsdp_model.train()
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

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred = self.fsdp_model(
                        **x.to('cuda'), labels=y.to('cuda'))

                loss = pred['loss'] / grad_accu_step
                loss.backward()
                mini_batch_loss += loss.detach()
                mini_batch_ppl += perplexity(pred['logits']
                                             [:, :-1, :], y[:, 1:], ignore_index=-100)

                if i % grad_accu_step == grad_accu_step - 1:
                    mini_batch_ppl /= grad_accu_step
                    param_norm = torch.nn.utils.get_total_norm(
                        self.fsdp_model.parameters())
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.fsdp_model.parameters(), self.clip_grad_max_norm)

                    dist.barrier()
                    dist.reduce(mini_batch_loss, 0, op=dist.ReduceOp.AVG)
                    dist.reduce(mini_batch_ppl, 0, op=dist.ReduceOp.AVG)

                    self.lr_scheduler.step()
                    self.optimizer.step()

                    self.step_cnt += 1
                    if self.rank == 0:
                        self.writer.add_scalars(
                            'norm', {
                                'param': param_norm.to_local(),
                                'grad': grad_norm.to_local(),
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
                        val_loss, val_ppl = self.validate(
                            self.val_dataloader)
                        if self.rank == 0:
                            self.writer.add_scalars(
                                'perplexity', {'val': val_ppl}, self.step_cnt)
                            self.writer.add_scalars(
                                'loss', {'val': val_loss}, self.step_cnt)

                        if self.save_dir is not None:
                            if len(self.top_k_list) == self.save_top_k and self.top_k_list[-1]['loss'] < val_loss:
                                # no need to save
                                pass
                            else:
                                save_path = os.path.join(
                                    self.save_dir, '{}_{}_{:.2f}'.format(epoch, self.step_cnt, val_loss))
                                if len(self.top_k_list) > 0:
                                    ckpt_path_to_remove = self.top_k_list[-1]['path']
                                    if len(self.top_k_list) == self.save_top_k:
                                        if self.rank == 0:
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
                                    'loss': val_loss,
                                })
                                self.top_k_list = sorted(
                                    self.top_k_list, key=lambda x: x['loss'])
                                self.save_checkpoint(save_path)

                                if self.rank == 0:
                                    print('model saved to {}'.format(save_path))

                    if self.rank == 0:
                        pbar.update(1)

    def train(self):
        max_len_steps = [0, 2048, 4096, 6144, 8192, 16384]
        batch_size_steps = [12, 6, 4, 3, 1]
        grad_acc_steps = [7, 14, 21, 28, 84]
        collate_fn_generator = CollateFn(TOKENIZER_NAME)
        total_steps_cnt_after_this_dataloader = 0

        for low, high, batch_size, grad_acc_step in zip(max_len_steps[:-1], max_len_steps[1:], batch_size_steps, grad_acc_steps):
            if self.rank == 0:
                print('max length: {}, batch size: {}, gradient accumulation steps: {}'.format(
                    high, batch_size, grad_acc_step))

            current_train_dataset = self.train_dataset.filter(
                lambda x: low < x['token_cnt'] <= high
            )
            current_train_dataset = current_train_dataset.sort(
                'token_cnt', reverse=False)

            current_val_dataset = self.val_dataset.filter(
                lambda x: low < x['token_cnt'] <= high
            )
            current_val_dataset = current_val_dataset.sort(
                'token_cnt', reverse=False)

            collate_fn_generator.max_length = high
            collate_fn = collate_fn_generator.collate_fn
            train_sampler = DistributedSampler(
                current_train_dataset, shuffle=False)
            self.train_dataloader = StatefulDataLoader(current_train_dataset, shuffle=False,
                                                       sampler=train_sampler, collate_fn=collate_fn, batch_size=batch_size, num_workers=4)
            val_sampler = DistributedSampler(
                current_val_dataset, shuffle=False)
            self.val_dataloader = StatefulDataLoader(current_val_dataset, shuffle=False,
                                                     sampler=val_sampler, collate_fn=collate_fn, batch_size=batch_size, num_workers=2)

            steps_in_current_train_loader = math.ceil(
                len(self.train_dataloader) / grad_acc_step)
            total_steps_cnt_after_this_dataloader += steps_in_current_train_loader
            if self.previous_ckpt_step > total_steps_cnt_after_this_dataloader:
                self.step_cnt += steps_in_current_train_loader
                continue
            else:
                self.train_on_dataloader(
                    train_sampler, grad_acc_step)

        save_path = os.path.join(self.save_dir, 'final_ckpt')
        self.save_checkpoint(save_path)
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
    parser.add_argument('--load_from_ckpt', default=None, type=str)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    main(
        save_frequency=args.save_frequency,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        save_top_k=args.save_top_k,
        clip_grad_max_norm=args.clip_grad_max_norm,
        load_from_ckpt=args.load_from_ckpt,
    )
