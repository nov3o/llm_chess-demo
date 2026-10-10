"""Export CHSM8 to ONNX for web/browser.html, and check it against PyTorch.

    python export_onnx.py --out onnx/        # writes chsm8.onnx (fp32) and chsm8_int8.onnx (browser)

Inputs : rows (1, T, 5) int64        BOS + one row per move (from the Rust tokenizer)
         text_states (1, L, 768) f32 EmbeddingGemma-300m token states of the prompt
         text_mask (1, L) bool       True for real prompt tokens
         use_text (1,) f32           0 = no prompt (pass L = 1 zeros), 1 = prompt
Outputs: src (1, 64), dst (1, 64), piece (1, 6), promo (1, 5): log-probs at the last position.
The browser computes text_states with transformers.js (onnx-community/embeddinggemma-300m-ONNX).
"""
import argparse
import os

import torch

import chsm8


class Exported(torch.nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.dec = dec

    def forward(self, rows, text_states, text_mask, use_text):
        d = self.dec
        x = d.embed(rows)
        rope = chsm8.rope_factors(rows.shape[1], rows.device)
        ctx, mask = d.text_proj(text_states), text_mask[:, None, None, :]
        for i, layer in enumerate(d.layers):
            j = i - (chsm8.LAYERS - chsm8.TEXT_LAYERS)
            if j >= 0:
                x = layer(x, rope, ctx, mask, torch.tanh(d.gate[j]) * use_text)
            else:
                x = layer(x, rope)
        h = d.norm(x[:, -1])
        return tuple(torch.log_softmax(d.head[k](h), -1) for k in chsm8.HEADS_OUT)


def export(dec, path):
    T, L = 9, 7
    args = (torch.zeros(1, T, 5, dtype=torch.long), torch.zeros(1, L, 768), torch.ones(1, L, dtype=torch.bool),
            torch.ones(1))
    args[0][0, 0, 0] = 1
    torch.onnx.export(Exported(dec).eval(), args, path, dynamo=False, opset_version=18,
                      input_names=["rows", "text_states", "text_mask", "use_text"],
                      output_names=list(chsm8.HEADS_OUT),
                      dynamic_axes={"rows": {1: "T"}, "text_states": {1: "L"}, "text_mask": {1: "L"}})


def variants(path, out):
    """Browser format: 8-bit weights (per output column scale), fp32 compute. Each MatMul weight becomes
    int8 -> DequantizeLinear -> MatMul, plain ops that ONNX Runtime Web runs on WebGPU and WebAssembly.
    Activations stay fp32: quantizing them (standard dynamic int8) breaks this model."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(path)
    inits = {i.name: i for i in m.graph.initializer}
    new_nodes = []
    for n in m.graph.node:
        w = inits.get(n.input[1]) if n.op_type == "MatMul" and len(n.input) > 1 else None
        if w is not None and len(w.dims) == 2:
            a = numpy_helper.to_array(w)
            scale = np.maximum(np.abs(a).max(0), 1e-12) / 127.0
            q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
            m.graph.initializer.remove(w)
            m.graph.initializer.extend([numpy_helper.from_array(q, w.name + "_q"),
                                        numpy_helper.from_array(scale.astype(np.float32), w.name + "_scale")])
            new_nodes.append(helper.make_node("DequantizeLinear", [w.name + "_q", w.name + "_scale"], [w.name],
                                              axis=1, name=w.name + "_dq"))
        new_nodes.append(n)
    del m.graph.node[:]
    m.graph.node.extend(new_nodes)
    onnx.save(m, os.path.join(out, "chsm8_int8.onnx"))


@torch.no_grad()
def check(model, out):
    """Every variant vs PyTorch: largest log-prob error over legal moves, and how often the top move differs."""
    import onnxruntime as ort
    import chsm8_tok
    games = ["", "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3",
             "1. d4 Nf6 2. c4 e6 3. Nc3 Bb4 4. e3 O-O 5. Bd3 d5 6. Nf3 c5 7. O-O dxc4 8. Bxc4 Nbd7",
             "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 8. c3 O-O 9. h3"]
    prompts = ["", "castle queenside and storm the enemy king with your pawns", "trade queens as early as possible"]
    sessions = {f: ort.InferenceSession(os.path.join(out, f)) for f in sorted(os.listdir(out)) if f.endswith(".onnx")}
    for name, s in sessions.items():
        worst, pworst, flips, n = 0.0, 0.0, 0, 0
        for g in games:
            rows, legal = chsm8_tok.analyze(g)
            for p in prompts:
                ref = model.score(g, p)
                states, mask = model._text(p) if p else (torch.zeros(1, 1, 768), torch.ones(1, 1, dtype=torch.bool))
                o = dict(zip(chsm8.HEADS_OUT, s.run(None, {
                    "rows": torch.tensor([rows]).numpy(), "text_states": states.cpu().float().numpy(),
                    "text_mask": mask.cpu().numpy(), "use_text": torch.tensor([1.0 if p else 0.0]).numpy()})))
                lp = {u: o["src"][0, r[1]] + o["dst"][0, r[2]] + o["piece"][0, r[3]] + o["promo"][0, r[4]] for u, _, r in legal}
                worst = max(worst, max(abs(lp[m["uci"]] - m["logp"]) for m in ref))
                pa = torch.softmax(torch.tensor([m["logp"] for m in ref]), 0)
                pb = torch.softmax(torch.tensor([float(lp[m["uci"]]) for m in ref]), 0)
                pworst = max(pworst, float((pa - pb).abs().max()))
                flips += max(lp, key=lp.get) != ref[0]["uci"]; n += 1
        mb = os.path.getsize(os.path.join(out, name)) / 2**20
        print(f"{name:20s} {mb:7.0f} MB   max |logp error| {worst:.4f}   max |prob error| {pworst:.4f}   top move differs in {flips}/{n} positions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="onnx")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m = chsm8.CHSM8.from_pretrained(device="cpu")
    path = os.path.join(a.out, "chsm8.onnx")
    export(m.decoder.float(), path)
    variants(path, a.out)
    check(m, a.out)
