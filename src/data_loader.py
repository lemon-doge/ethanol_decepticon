import torch
from settings import data_path, device 
from tokenizer import CharTokenizer

class DataLoader():
    def __init__(self, tokenizer: CharTokenizer, mode: str, data_path = data_path, device = device):
        self.tokenizer = tokenizer

        if mode == "train":
            with open(data_path + "/train.txt", 'r', encoding='utf-8') as f:
                self.data = self.tokenizer.encode(f.read())
        elif mode == "val":
            with open(data_path + "/val.txt", 'r', encoding='utf-8') as f:
                self.data = self.tokenizer.encode(f.read())
        else:
            raise RuntimeError(f"unknown mode: {mode}")

    def sample_batch(self, batch_size:int, context_len:int) -> [torch.tensor, torch.tensor]:
        start_idx = torch.randint(0, len(self.data) - context_len, (batch_size,))
        x = torch.vstack([self.data[s: s+context_len] for s in start_idx]) # batch_size x context_len
        y = torch.vstack([self.data[s+1: s+1+context_len] for s in start_idx]) # batch_size x context_len
        return x.to(device), y.to(device)


