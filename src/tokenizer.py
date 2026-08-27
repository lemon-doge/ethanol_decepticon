import torch
from settings import data_path

## a simple character level tokenizer 

class CharTokenizer():
    def __init__(self, data_path=data_path):
        with open(data_path + "/allsektorgaza.txt", 'r', encoding='utf-8') as f:
             alltext = f.read()
        self.unique = sorted(list(set(alltext)))
        self.c2i = {c:i for i,c in enumerate(self.unique)}
        self.i2c = {i:c for i,c in enumerate(self.unique)}
        self.vocab_size = len(self.unique)

    def encode(self, text:str) -> torch.tensor:
        return torch.tensor([self.c2i[c] for c in text])

    def decode(self, seq:torch.tensor) -> str:
        return ''.join([self.i2c[int(i)] for i in seq])

    
