"""Local CHSM8 backend for web/index.html.

    pip install -r requirements.txt
    python server.py            # http://localhost:8000, device auto: cuda > mps > cpu
"""
import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chsm8 import CHSM8


class Query(BaseModel):
    moves: str = ""   # game so far, any notation
    prompt: str = ""  # style description for the side to move ("" = no prompt)


def create_app(model: CHSM8) -> FastAPI:
    app = FastAPI()
    # The Vercel-hosted page calls this server on your machine.
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/info")
    def info():
        return {"model": "LegumMagister/chsm8", "device": str(model.device)}

    @app.post("/api/score")
    def score(q: Query):
        try:
            return {"moves": model.score(q.moves, q.prompt)}
        except ValueError as e:
            raise HTTPException(400, str(e))

    app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True))
    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: best available)")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    m = CHSM8.from_pretrained(device=a.device)
    print(f"CHSM8 on {m.device}: open http://localhost:{a.port}")
    uvicorn.run(create_app(m), host="127.0.0.1", port=a.port)
