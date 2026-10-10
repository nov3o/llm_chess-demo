"""CHSM8 inference: a chess move model conditioned on a free-text style description.

    from chsm8 import CHSM8
    m = CHSM8.from_pretrained()                     # LegumMagister/chsm8, device auto: cuda > mps > cpu
    m.score("1. e4 e5 2. Nf3", "go for the Sicilian Defense")
    # -> [{'uci': 'b8c6', 'san': 'Nc6', 'logp': -0.9}, ...] for every legal move, best first

The decoder (12 layers, width 960) reads the game as one row per move. The prompt is encoded by a frozen
EmbeddingGemma-300m; its token states are cross-attended by the top 6 layers. With no prompt the model plays
its pretrained policy. Move text in any notation is parsed by the Rust tokenizer (chsm8_tok).
"""
import math
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch import nn

import chsm8_tok  # pip install "git+https://huggingface.co/LegumMagister/chsm8-tokenizer"

D, HEADS, LAYERS, FF, TEXT_LAYERS, MAX_ROWS = 960, 15, 12, 3520, 6, 512
HEADS_OUT = {"src": 64, "dst": 64, "piece": 6, "promo": 5}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Attention(nn.Module):
    def __init__(self, qk_norm):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(D, D) for _ in range(4))
        self.hd = D // HEADS
        if qk_norm:
            self.q_norm, self.k_norm = RMSNorm(self.hd), RMSNorm(self.hd)
        self.qk_norm = qk_norm

    def forward(self, x, ctx=None, mask=None, rope=None):
        B, T, _ = x.shape
        kv = x if ctx is None else ctx
        q = self.q_proj(x).view(B, T, HEADS, self.hd)
        k = self.k_proj(kv).view(B, kv.shape[1], HEADS, self.hd)
        v = self.v_proj(kv).view(B, kv.shape[1], HEADS, self.hd)
        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:  # interleaved RoPE on self-attention
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                             attn_mask=mask, is_causal=ctx is None)
        return self.out_proj(out.transpose(1, 2).reshape(B, T, D))


def rope_factors(T, device):
    inv = 1.0 / (10000 ** (torch.arange(0, D // HEADS, 2, device=device, dtype=torch.float32) / (D // HEADS)))
    ang = torch.arange(T, device=device, dtype=torch.float32)[:, None] * inv
    return ang.cos()[None, :, None, :], ang.sin()[None, :, None, :]


def apply_rope(x, cos, sin):
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(D, FF, bias=False), nn.Linear(D, FF, bias=False)
        self.down_proj = nn.Linear(FF, D, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, text):
        super().__init__()
        self.self_attn, self.n1 = Attention(qk_norm=True), RMSNorm(D)
        self.ff, self.n3 = SwiGLU(), RMSNorm(D)
        self.no_text = nn.Parameter(torch.zeros(D))  # the pretrained policy's constant cross-attention term
        if text:
            self.cross_attn, self.n2 = Attention(qk_norm=False), RMSNorm(D)

    def forward(self, x, rope, ctx=None, mask=None, gate=None):
        n = self.n1(x)
        x = x + self.self_attn(n, rope=rope) + self.no_text
        if ctx is not None:
            x = x + gate * self.cross_attn(self.n2(x), ctx, mask)
        return x + self.ff(self.n3(x))


class Embed(nn.Module):
    def __init__(self):
        super().__init__()
        self.kind_emb, self.src_emb, self.dst_emb = nn.Embedding(4, D), nn.Embedding(64, D), nn.Embedding(64, D)
        self.piece_emb, self.promo_emb = nn.Embedding(6, D), nn.Embedding(5, D)

    def forward(self, rows):  # rows (B, T, 5): kind, from, to, piece, promotion
        kind, src, dst, piece, promo = rows.unbind(-1)
        move = self.src_emb(src) + self.dst_emb(dst) + self.piece_emb(piece) + self.promo_emb(promo)
        return torch.where((kind == 3)[..., None], move, self.kind_emb(kind))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = Embed()
        self.layers = nn.ModuleList(Layer(text=i >= LAYERS - TEXT_LAYERS) for i in range(LAYERS))
        self.norm = RMSNorm(D)
        self.head = nn.ModuleDict({k: nn.Linear(D, n) for k, n in HEADS_OUT.items()})
        self.text_proj = nn.Sequential(nn.Linear(768, D), RMSNorm(D))  # EmbeddingGemma token states -> width
        self.gate = nn.Parameter(torch.zeros(TEXT_LAYERS))

    def forward(self, rows, text_states=None, text_mask=None):
        """rows (B, T, 5) -> log-probs of each head at the last position."""
        x = self.embed(rows)
        rope = rope_factors(rows.shape[1], rows.device)
        ctx = mask = None
        if text_states is not None:
            ctx, mask = self.text_proj(text_states), text_mask[:, None, None, :]
        for i, layer in enumerate(self.layers):
            j = i - (LAYERS - TEXT_LAYERS)
            x = layer(x, rope, *((ctx, mask, torch.tanh(self.gate[j])) if j >= 0 and ctx is not None else ()))
        h = self.norm(x[:, -1])
        return {k: F.log_softmax(head(h).float(), -1) for k, head in self.head.items()}


def convert(ckpt):
    """Released training checkpoint -> Decoder state dict."""
    sd = {}
    for k, v in ckpt["chess"].items():
        if (".cross_attn." in k or ".n2." in k) and int(k.split(".")[1]) < LAYERS - TEXT_LAYERS:
            continue  # bottom layers never see text: only their constant no-text term is kept
        if k.startswith("head.kind_head"):
            continue  # start/end-of-game head, unused for move choice
        sd[k.replace("_head.", ".")] = v
    for i, t in ckpt["frozen_dummy"].items():
        sd[f"layers.{int(i)}.no_text"] = t.reshape(-1)
    sd["text_proj.0.weight"], sd["text_proj.0.bias"] = ckpt["cond"]["proj.0.weight"], ckpt["cond"]["proj.0.bias"]
    sd["text_proj.1.weight"], sd["gate"] = ckpt["cond"]["proj.1.weight"], ckpt["cond"]["gate"]
    return sd


class CHSM8:
    def __init__(self, decoder, encoder, tokenizer, device):
        self.decoder, self.encoder, self.tok, self.device = decoder, encoder, tokenizer, device

    @classmethod
    def from_pretrained(cls, repo="LegumMagister/chsm8", filename="model.pt",
                        encoder="google/embeddinggemma-300m", device=None):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, AutoTokenizer
        device = torch.device(device) if device else pick_device()
        ckpt = torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=True)
        dec = Decoder()
        dec.load_state_dict(convert(ckpt), strict=True)
        enc = AutoModel.from_pretrained(encoder, dtype=torch.bfloat16)  # as in training
        tok = AutoTokenizer.from_pretrained(encoder)
        return cls(dec.to(device).eval(), enc.to(device).eval(), tok, device)

    @lru_cache(maxsize=64)
    def _text(self, prompt):
        enc = self.tok([prompt], truncation=True, max_length=128, return_tensors="pt").to(self.device)
        states = self.encoder(**enc).last_hidden_state.float()
        return states, enc["attention_mask"].bool()

    @torch.no_grad()
    def score(self, moves="", prompt=""):
        """Game so far (any notation) + style prompt -> every legal move with its log-probability."""
        rows, legal = chsm8_tok.analyze(moves)
        if not legal:
            return []
        x = torch.tensor([rows[-MAX_ROWS:]], dtype=torch.long, device=self.device)
        text = self._text(prompt.strip()) if prompt.strip() else (None, None)
        lp = self.decoder(x, *text)
        idx = torch.tensor([r for _, _, r in legal], device=self.device)
        s = lp["src"][0, idx[:, 1]] + lp["dst"][0, idx[:, 2]] + lp["piece"][0, idx[:, 3]] + lp["promo"][0, idx[:, 4]]
        out = [{"uci": u, "san": san, "logp": float(v)} for (u, san, _), v in zip(legal, s.tolist())]
        return sorted(out, key=lambda m: -m["logp"])
