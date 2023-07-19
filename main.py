import numpy as np
import pandas as pd
import torch
import csv
import os, sys
from os.path import join, abspath, dirname
from datetime import datetime
from transformers import RobertaTokenizer, T5ForConditionalGeneration, get_scheduler, AdamW, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from code_datasets import CodeContestDataset

import argparse
import jsonlines
import json

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

import torch.nn as nn


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

SUPPORT_MODELS = ['microsoft/CodeGPT-small-py', 'EleutherAI/gpt-j-6B', 'EleutherAI/gpt-neo-1.3B', 'EleutherAI/gpt-neo-125M', 'EleutherAI/gpt-neo-2.7B', 'Salesforce/codet5-small', 'Salesforce/codet5-large', 'Salesforce/codet5p-770m-py']

def log(string, args):
    if args.world_size == 1 or torch.distributed.get_rank() == 0:
        print(string, flush=True)

def construct_generation_args():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--use_empty_cache", action="store_true")
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--model_name", type=str, default='Salesforce/codet5p-770m-py', choices=SUPPORT_MODELS)
    parser.add_argument("--ckpt_pathname", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='../baseline/data/codecontest/')
    parser.add_argument("--out_dir", type=str, default='./out/')
    parser.add_argument("--train_basename", type=str, default='codecontest_train.jsonl')
    parser.add_argument("--valid_basename", type=str, default='codecontest_valid.jsonl')
    parser.add_argument("--test_basename", type=str, default='codecontest_test.jsonl')
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--max_sequence_length", type=int, default=512, help='학습 입력 Sequence 길이 제한(Context+Label)')
    parser.add_argument("--max_generation_length", type=int, default=512) 
    parser.add_argument("--train_data_size", type=int, default=-1) # Full
    parser.add_argument("--test_data_size", type=int, default=-1) # Full

    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", type=str2bool, default=False)

    parser.add_argument("--preprocess_text", type=str2bool, default=True)
    
    parser.add_argument("--task_name", type=str, default='noramalgen')

    parser.add_argument("--seed", type=int, default=34, help="random seed for initialization")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--decay_rate", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)

    parser.add_argument("--mode", default='finetune', choices=['finetune', 'wte'])

    parser.add_argument("--dataset_type", type=str, default='codecontest',
        choices=[
            'humaneval',
            'apps',
            'codecontest',
        ])
    parser.add_argument("--test_type", type=str, default='valid')
    parser.add_argument("--generation_max_length", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=1)

    parser.add_argument("--print_train_metric", type=str2bool, default=True)
    parser.add_argument("--save_test_inference", type=str, default='./pred/second_gen_large.tsv')
    parser.add_argument("--save_model", type=str2bool, default=True)
    parser.add_argument("--save_model_path", type=str, default=None)

    parser.add_argument("--do_sample", type=str2bool, default=True)
    parser.add_argument("--precision", type=str, default='mp', choices=['fp16', 'fp32', 'mp'])
    parser.add_argument("--optimizer", type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument("--scheduler", type=str, default='CosineScheduleWithWarmUp', choices=['ExponentialLR', 'TriStageLRScheduler', 'ReduceLROnPlateauScheduler', 'CosineScheduleWithWarmUp'])

    # parser.add_argument("--num_labels", type=int, default=2)
    # parser.add_argument("--label_smoothing", type=float, default=0.0)

    parser.add_argument('--local_rank', type=int, default=0, help='local rank passed from distributed launcher')

    # parser.add_argument('--focal_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--ldam_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--new_cb_usage', type=str2bool, default=False, help='boolean for usage of focal loss')
    # parser.add_argument('--new_cb_type', type=str, default='focal', help='Normal for Normal Focal loss, Class for Class Balanced Focal Loss')

    # parser.add_argument('--focal_type', type=str, default='Normal', help='Normal for Normal Focal loss, Class for Class Balanced Focal Loss')
    # parser.add_argument('--focal_gamma', type=float, default=2., help='gamma for focal loss')
    # parser.add_argument('--focal_alpha', type=float, default=0.25, help='alpha for usage of focal loss')

    args = parser.parse_args()

    args.use_pad_sequence_max = False

    assert args.accumulation_steps > 0

    assert not (args.mode == 'wte' and args.precision == 'fp16'), "WTE 모드는 FP16에서 실행되지 않습니다!!!"

    set_seed(args)

    return args


def main():
    
    args = construct_generation_args()
    def average_gradients(model, args):
        size = float(args.world_size)
        for name, param in model.named_parameters():
            if param.grad == None:
                continue
            torch.distributed.all_reduce(param.grad.data, op=torch.distributed.ReduceOp.SUM)
            param.grad.data /= size

    def get_checkpoint(epoch_idx, test_ppl, train_ppl, args, test_set):
        ckpt_name = ''
        if args.world_size == 1:
            ckpt_name = "epoch_{}_test_{}_train_{}.ckpt".format(epoch_idx, test_ppl, train_ppl)
        elif args.world_size > 1:
            ckpt_name = "epoch_{}_test_{}_train_{}_{}.ckpt".format(epoch_idx, test_ppl, train_ppl, torch.distributed.get_rank())

        embedding = None
        if args.world_size > 1:
            embedding = model.module.state_dict()
        else:
            embedding = model.state_dict()
        return {'embedding': embedding,
                'test_ppl': test_ppl,
                'test_size': len(test_set),
                'ckpt_name': ckpt_name,
                'time': datetime.now(),
                'args': args}

    def save(best_ckpt, args):
        ckpt_name = best_ckpt['ckpt_name']
        path = get_save_path(args)
        os.makedirs(path, exist_ok=True)
        torch.save(best_ckpt, join(path, ckpt_name))

    def get_task_name(args):
        names = [args.model_name,
                args.mode,
                args.dataset_type]
        return "_".join(names)

    def get_save_path(args):
        if args.save_model_path != None:
            return join(args.save_model_path, get_task_name(args))

        return join(args.out_dir, args.model_name, args.mode, get_task_name(args))
    
    # 모델과 토크나이저 초기화
    model_name = args.model_name
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    for name, param in model.named_parameters():
        param.requires_grad = True
    
    
    #데이터셋
    print("데이터 읽기 시작")    
    if args.eval_only == False:
        train_dataset = CodeContestDataset(args.data_dir + args.train_basename, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    elif args.eval_only == True:
        train_loader = None
    if args.test_type == 'valid':
        valid_set = CodeContestDataset(args.data_dir + args.valid_basename, tokenizer)
        test_loader = DataLoader(valid_set, batch_size=args.test_batch_size)
    else:
        test_set = CodeContestDataset(args.data_dir + args.test_basename, tokenizer)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size) 
    
    # Accelerator 초기화
    accelerator = Accelerator()
    
    def evaluate(epoch_idx, model, tokenizer, test_loader, args):
        model.eval()

        eval_b_cnt=0
        eval_loss=0
        num_test_steps = len(test_loader)
        progress_bar = tqdm(range(num_test_steps))
        with torch.no_grad():
            log(f"### START TEST ###", args)
            torch.cuda.empty_cache()

            predictions = []
            contexts = []
            task_ids = []
            for batch in test_loader:
                context = batch["prompt"]
                input_ids = batch["input_ids"].squeeze(1)
                attention_mask = batch["attention_mask"].squeeze(1)
                labels = batch["labels"].squeeze(1)
                task_id = batch["task_id"]
                if args.world_size > 1:
                    predictions += model.module.generate(input_ids=input_ids.to(model.device).long(),
                                                        attention_mask=attention_mask.to(model.device).long(), 
                                                        do_sample=True,
                                                        top_k=50,
                                                        top_p=0.92,
                                                        temperature=0.9,
                                                        max_length=args.max_generation_length)
                else:
                    predictions += model.generate(input_ids=input_ids.to(model.device).long(),
                                                        attention_mask=attention_mask.to(model.device).long(), 
                                                        do_sample=True,
                                                        top_k=50,
                                                        top_p=0.92,
                                                        temperature=0.9,
                                                        max_length=args.max_generation_length)
                task_ids += task_id
                outputs = model(input_ids=input_ids.to(model.device).long(), attention_mask=attention_mask.to(model.device).long(), labels = labels.to(model.device).long(), decoder_input_ids = labels.to(model.device).long())                
                _loss = outputs.loss
                for context_row in context:
                    contexts.append(context_row.replace("\n", "\\n"))
                # for labels_row in labels:
                #     references.append(labels_row.replace("\n", "\\n"))
                eval_b_cnt+=1
                eval_loss += _loss.item()
                progress_bar.update(1)

            eval_loss = eval_loss/eval_b_cnt
            perplexity = torch.exp(torch.tensor(eval_loss))

            log(f"Test Epoch: {epoch_idx} Loss: {eval_loss} perplexity: {perplexity}\n", args)
        if args.world_size == 1 or (args.save_test_inference != None and torch.distributed.get_rank() == 0):
            with open(args.save_test_inference, 'a') as f:
                wr = csv.writer(f, delimiter="\t")
                for idx in range(len(predictions)):
                    wr.writerow([epoch_idx, int(task_ids[idx]), tokenizer.decode(predictions[idx], skip_special_tokens = True).replace('\n', '\\n').replace('\t','\\t').encode('utf-8').decode('utf-8'), contexts[idx].replace('\n', '\\n').replace('\t','\\t').encode('utf-8').decode('utf-8')])

        return eval_loss, perplexity
        

    def train(model, train_loader, test_loader, args):
        optimizer = AdamW(model.parameters(), 
                            lr=args.lr, 
                            weight_decay = args.weight_decay)                
        train_steps = len(train_loader) * args.max_epochs // args.accumulation_steps
        warmup_steps = int(train_steps * 0.1)
        lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=warmup_steps, 
                                                        num_training_steps=train_steps)
        # 훈련 데이터셋 로드

        # 훈련 루프
        for epoch in range(args.max_epochs):
            num_training_steps = len(train_loader)
            progress_bar = tqdm(range(num_training_steps))
            total_loss=0
            train_b_cnt=0
            torch.cuda.empty_cache()
            model.train()
            for steps, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].squeeze(1)
                attention_mask = batch["attention_mask"].squeeze(1)
                labels = batch["labels"].squeeze(1)

                outputs = model(input_ids=input_ids.to(model.device).long(), attention_mask=attention_mask.to(model.device).long(), labels = labels.to(model.device).long(), decoder_input_ids = labels.to(model.device).long())
                
                loss = outputs.loss
                loss = loss / args.accumulation_steps
                total_loss += loss.item()
                accelerator.backward(loss)
                if args.world_size > 1:
                    average_gradients(model.module, args)
                if (steps+1) % args.accumulation_steps == 0:
                    if args.use_empty_cache == True:
                        torch.cuda.empty_cache()
                    
                    optimizer.step()
                    optimizer.zero_grad()
                
                    lr_scheduler.step()
                
                progress_bar.update(1)
                train_b_cnt+=1
                if args.print_train_metric and steps%10==0:
                    log(f"Train LR: {lr_scheduler.get_last_lr()[0]:.10f} Epoch {epoch} Step: {steps} Loss: {total_loss/train_b_cnt:.5f} perplexity: {torch.exp(torch.tensor(total_loss/train_b_cnt)):.5f}", args)
            
            if train_b_cnt != 0:
                total_loss = total_loss/train_b_cnt
            train_ppl = torch.exp(torch.tensor(total_loss))
            if args.world_size == 1 or torch.distributed.get_rank() == 0:
                log("Train LR: {} Epoch {} Loss: {} perplexity: {}".format(lr_scheduler.get_last_lr(),epoch, total_loss, train_ppl), args)
            
            test_ppl = None
            if args.no_eval != True:
                _, test_ppl = evaluate(epoch, model, tokenizer, test_loader, args)

            best_ckpt = get_checkpoint(epoch, test_ppl, train_ppl, args, test_loader)
            if args.save_model == True:
                save(best_ckpt, args)
            torch.cuda.empty_cache()
        return best_ckpt

    # 모델과 토크나이저를 가속화된 디바이스에 래핑합니다.
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.to(accelerator.device)
    if args.world_size == 1 or (args.save_test_inference != None and torch.distributed.get_rank() == 0):
        with open(args.save_test_inference, 'w') as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(['epoch', 'task_id', 'completion', 'prompt'])
    if not args.eval_only:
        train(model, train_loader, test_loader, args)
    else:
        evaluate(model, test_loader, args)   


if __name__ == '__main__':
    main()