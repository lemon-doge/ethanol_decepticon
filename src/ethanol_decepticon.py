import torch
import torch.nn as nn
from settings import device

## small decoder-only model to generate infinite Sektor Gaza

## hyperprams for generation
temp = 0.8  # todo watch what happens wit 0.2
top_k = 10  # todo try 10, 20


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
        s /= d_k**0.5
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
            seq = input_seq.clone().detach()

            for _ in range(max_new_tokens):
                logits, _ = self.forward(
                    seq[:, -self.context_len :]
                )  # B x CL x vocab_size
                logits = logits[:, -1, :]  # B x vocab_size — only the next-token prediction

                ## add temperature
                logits = logits / temp
                ## add top-k sampling
                values, _ = torch.topk(logits, k=top_k, dim=-1)  # B x top_k
                # sorted descendingly -> values[:, [-1]] is the k-th largest (cutoff)
                logits[logits < values[:, [-1]]] = float("-inf")

                probabilities = nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                seq = torch.cat([seq, next_token], dim=1)

            return seq
