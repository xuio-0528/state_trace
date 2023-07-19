import torch
from torch.utils.data import Dataset
import json
from accelerate import Accelerator

class CodeContestDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.data = []
        self.tokenizer = tokenizer

        with open(file_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.data.append(example)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        task_id = example["task_id"]
        prompt = example["prompt"].encode('utf-8').decode("utf-8")
        solution = example["solution"].encode('utf-8').decode("utf-8")

        # 토크나이징
        inputs = self.tokenizer(
            prompt,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.squeeze()
        attention_mask = inputs.attention_mask.squeeze()
        target = self.tokenizer(
            solution,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        labels = target.input_ids.long()
        return {
            "prompt":prompt,
            "task_id" : task_id,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels" : labels
        }