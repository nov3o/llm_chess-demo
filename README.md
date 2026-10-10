# CHSM8: a chess model that plays in the style you describe

CHSM8 is a 210.5M-parameter chess transformer that picks its moves from a free-text description of a playing style,
such as *"castle queenside and storm the enemy king"*, *"go for the Sicilian Defense"* or *"solid, safe, risk-free
chess"*. It plays without search and only legal moves.

**Project page and demos:** the `web/` folder (deployed on Vercel). It has the project page, an in-browser demo and
a demo that runs on your own GPU.
**Models and data:** [huggingface.co/LegumMagister](https://huggingface.co/LegumMagister).

This repository holds inference code only: the model, the two demos and the browser export.
The training code will be released later.

## Play on your GPU (PyTorch)

```bash
pip install -r requirements.txt   # includes the Rust tokenizer, which needs a Rust toolchain (https://rustup.rs)
python server.py                 # picks cuda, then mps, then cpu; open http://localhost:8000/play.html, choose "My server"
```

The prompt encoder is [`google/embeddinggemma-300m`](https://huggingface.co/google/embeddinggemma-300m), which is
gated: accept its license on Hugging Face and run `hf auth login` once.
The Vercel-hosted `play.html` can also use this local server: choose "My server" (default `http://localhost:8000`) and press "Test connection".

## Play in the browser (no server)

`web/play.html` runs everything in the tab with [ONNX Runtime Web](https://onnxruntime.ai) (WebGPU, falling back to
WebAssembly) and [transformers.js](https://github.com/huggingface/transformers.js):

| step | runs as | download |
|---|---|---|
| moves → model rows | Rust tokenizer compiled to WebAssembly (`web/pkg`) | 1.3 MB |
| prompt → token states | EmbeddingGemma-300m, 8-bit ([`onnx-community/embeddinggemma-300m-ONNX`](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX)) | 295 MB |
| rows + prompt → move probabilities | CHSM8 decoder with 8-bit weights ([`onnx/model_int8.onnx`](https://huggingface.co/LegumMagister/chsm8/tree/main/onnx)) | 183 MB |

Both files are cached by the browser after the first visit. The browser version stores weights in 8 bits and
computes in fp32. Against the full-precision PyTorch model, move probabilities differ by at most about 1
percentage point from the decoder and about 6 from the 8-bit text encoder (an exact 1.2 GB encoder is selectable).
On our test positions the top move matched in all but one case, where the two best moves were nearly tied.

To serve the pages locally without the Python server: `python -m http.server -d web`.

## Use the model from Python

```python
from chsm8 import CHSM8

model = CHSM8.from_pretrained()                    # downloads LegumMagister/chsm8
moves = model.score("1. e4 e5 2. Nf3", "castle queenside and storm the enemy king with your pawns")
print(moves[:3])                                   # [{'uci': 'b8c6', 'san': 'Nc6', 'logp': ...}, ...]
```

`score` takes the game so far in any notation (SAN, LAN, UCI, PGN with numbers and comments) and returns every legal
move with its log-probability, best first. An empty prompt gives the model's own play without a style.
`chsm8.py` is the whole model in about 180 lines; it reproduces the training code's outputs to about 1e-4.

## Tokenizer

Moves are turned into CHSM8 tokens by [chsm8-tokenizer](https://huggingface.co/LegumMagister/chsm8-tokenizer), a small
Rust crate that reads any common notation (SAN, LAN, UCI, PGN with numbers and comments). The server uses its Python
module; `web/pkg` is its prebuilt WebAssembly build, copied from that repository's `wasm/` folder.

## Rebuild the browser model

```bash
python export_onnx.py --out onnx/   # fp32 and 8-bit ONNX, plus a comparison against PyTorch
```

## Reuse and reproduce

You are welcome to reproduce our results, build on the models, and use the datasets and labels in your own work.
Weights, datasets with every training label (including the raw outputs of both teacher LLMs), prompts and
configurations are on Hugging Face; the technical report documents every stage.

| artefact | license |
|---|---|
| this code | MIT |
| chsm8, chsm8-sft, chsm8-pt, player models | MIT |
| chsm8-labeler-gemma3-1b (fine-tuned Gemma 3) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) |
| labels by Gemma-4-31B and Apertus-v1.5-70B, labelled post-training set | Apache-2.0, like both teachers |
| Lichess games | CC0 |
| EmbeddingGemma-300m (downloaded at run time, not redistributed) | Gemma Terms of Use |

## Citation

```bibtex
@techreport{badanin2026chsm8,
  title  = {CHSM8: A Chess Model That Plays in the Style You Describe},
  author = {Badanin, Ilia},
  year   = {2026},
  url    = {https://github.com/nov3o/llm_chess-demo}
}
```
