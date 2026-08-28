import torch
from tokenizer import CharTokenizer
from ethanol_decepticon import EthanolDecepticon
from settings import device, seed


torch.manual_seed(seed)

ckpt = torch.load("checkpoints/ethanol_decepticon.pt", map_location=device)

model = EthanolDecepticon(
    ckpt["vocab_size"],
    ckpt["context_len"],
    ckpt["d_model"],
    ckpt["d_k"],
    ckpt["d_ff"],
    ckpt["p_drop"],
    ckpt["n_heads"],
    ckpt["n_layers"],
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

tokenizer = CharTokenizer()
input_seq = tokenizer.encode("[Текст песни «Зелёный Слизень»]").to(device)
print(tokenizer.decode(model.generate(input_seq, 1000).tolist()))
