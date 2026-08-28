import torch
import torch.nn as nn
from data_loader import DataLoader
from tokenizer import CharTokenizer
from ethanol_decepticon import EthanolDecepticon
from settings import device, seed

## small decoder-only model to generate infinite Sektor Gaza

## hyperparams
torch.manual_seed(seed)

batch_size = 64
context_len = 256
lr = 3e-4
train_iters = 5_000
eval_interval = 500
eval_iters = 200
d_model = 64
d_ff = 4 * d_model
n_heads = 4
d_k = int(d_model / n_heads)
n_layers = 2
p_drop = 0.2


## model definition
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size

model = EthanolDecepticon(
    vocab_size,
    context_len,
    d_model,
    d_k,
    d_ff,
    p_drop,
    n_heads,
    n_layers,
).to(device)

## training cycle
train_dataloader = DataLoader(tokenizer, mode="train")
val_dataloader = DataLoader(tokenizer, mode="val")


def estimate_loss():
    with torch.no_grad():
        model.eval()

        train_losses = []
        val_losses = []
        for _ in range(eval_iters):
            ## estimate train loss
            x, y = train_dataloader.sample_batch(batch_size, context_len)
            _, loss = model.forward(x, y)
            train_losses.append(loss.item())

            ## estimate val loss
            x, y = val_dataloader.sample_batch(batch_size, context_len)
            _, loss = model.forward(x, y)
            val_losses.append(loss.item())

        model.train()

    return torch.tensor(train_losses).mean(), torch.tensor(val_losses).mean()


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(train_iters):
    if i % eval_interval == 0:
        train_loss, val_loss = estimate_loss()
        print(f"i: {i}, train_loss: {train_loss}, val_loss: {val_loss}")

    x, y = train_dataloader.sample_batch(batch_size, context_len)

    logits, loss = model.forward(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


## saving the model
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "step": i,
    # everything needed to reconstruct the model:
    "vocab_size": vocab_size,
    "context_len": context_len,
    "d_model": d_model,
    "d_k": d_k,
    "d_ff": d_ff,
    "p_drop": p_drop,
    "n_heads": n_heads,
    "n_layers": n_layers,
}
torch.save(checkpoint, "checkpoints/ethanol_decepticon.pt")
