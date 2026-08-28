import torch
import torch.nn as nn
from data_loader import DataLoader
from tokenizer import CharTokenizer
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
d_model = 192
d_ff = 4 * d_model
n_heads = 6
d_k = int(d_model / n_heads)
n_layers = 6
p_drop = 0.2


## model definition
tokenizer = CharTokenizer()
vocab_size = tokenizer.vocab_size


class AttentionHead(nn.Module):
    def __init__(
        self,
        context_len: int,
        d_model: int,
        d_k: int,
        p_drop: float,
    ):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_k, bias=False)
        self.WK = nn.Linear(d_model, d_k, bias=False)
        self.WV = nn.Linear(d_model, d_k, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_len, context_len)))
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        x.shape = batch_size x context_len x d_model
        """

        q = self.WQ.forward(x)
        k = self.WK.forward(x)
        v = self.WV.forward(x)

        c_len = x.shape[1]
        d_k = k.shape[2]

        ## B x CL x d_k @ B x d_k x CL -> B x CL x CL - hoping for the broadcast
        s = q @ k.permute(0, 2, 1)
        s /= d_k**-0.5
        ## B x CL x CL -> B x CL x CL
        s = s.masked_fill(self.tril[:c_len, :c_len] == 0, float("-inf"))
        s = nn.functional.softmax(s, dim=2)

        s = self.dropout(s)  # <- why here ?

        # B x CL x CL @ B x CL x d_k -> B x CL x d_k - again hoping for the broadcast
        return s @ v


class MHSA(nn.Module):
    def __init__(
        self,
        context_len: int,
        d_model: int,
        d_k: int,
        p_drop: float,
        n_heads: int,
    ):
        super().__init__()

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    context_len,
                    d_model,
                    d_k,
                    p_drop,
                )
                for _ in range(n_heads)
            ]
        )
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x: torch.tensor) -> torch.tensor:
        res = torch.cat(
            [head.forward(x) for head in self.heads], dim=-1
        )  # B x CL x d_model
        res = self.linear.forward(res)
        return self.dropout(res)


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        p_drop: float,
    ):
        super().__init__()

        self.ff = nn.Sequential(
            # creating a large feature space - processing the information in modified embeddings after self attention
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            # projects back to the original stream
            nn.Linear(d_ff, d_model),
            nn.Dropout(p_drop),
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.ff.forward(x)  # B x CL x d_model


class Block(nn.Module):
    def __init__(
        self,
        context_len: int,
        d_model: int,
        d_k: int,
        d_ff: int,
        p_drop: float,
        n_heads: int,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.mhsa = MHSA(context_len, d_model, d_k, p_drop, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, p_drop)

    def forward(self, x: torch.tensor) -> torch.tensor:
        ## pre layer norm
        x = x + self.mhsa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class EthanolDecepticon(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_len: int,
        d_model: int,
        d_k: int,
        d_ff: int,
        p_drop: float,
        n_heads: int,
        n_layers: int,
    ):
        super().__init__()

        self.context_len = context_len

        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pe = nn.Embedding(context_len, d_model)

        self.blocks = nn.Sequential(
            *[
                Block(context_len, d_model, d_k, d_ff, p_drop, n_heads)
                for _ in range(n_layers)
            ]
        )
        self.lf_f = nn.LayerNorm(d_model)

        ## creates logits
        self.lin_f = nn.Linear(d_model, vocab_size)

        ## loops through every child module returned by .children() and applies your argument function to it
        self.apply(self._init_weights)

    # why this is better ? - says its a golden standard for std
    def _init_weights(self, module: torch.nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        else:
            pass

    def forward(
        self, input_seq: torch.tensor, targets: torch.tensor = None
    ) -> [torch.tensor, torch.tensor]:
        """
        input_seq - integer input sequence tensor: B x CL
        targets - sequence of true next tokens: B x CL
        """

        bs, cl = input_seq.shape
        assert bs == targets.shape[0] and cl == targets.shape[1]

        res = self.embeddings(input_seq) + self.pe(
            torch.arange(cl, device=device)
        )  # B x CL -> B x CL x d_model

        res = self.blocks(res)  # B x CL x d_model -> B x CL x d_model

        res = self.lf_f(res)  # B x CL x d_model -> B x CL x d_model

        logits = self.lin_f(res)  # B x CL x d_model -> B x CL x vocab_size

        loss = (
            None
            if targets is None
            else nn.functional.cross_entropy(
                logits.view(bs * cl, -1),
                targets.view(bs * cl),
            )
        )  ## a scalar because mean by batch and cl dimensions is taken by default

        return logits, loss

    def generate(self, input_seq: torch.tensor, max_new_tokens: int):
        """
        input_seq - integer input sequence tensor inititally of shape: B x CL
        """
        with torch.no_grad():
            seq = input_seq.copy().detach()

            for _ in range(max_new_tokens):
                logits, _ = self.forward(seq[: -self.context_len :])
                probabilities = nn.functional.softmax(logits[:, -1, :], dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                seq = torch.cat([seq, next_token], dim=1)

            return seq


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
