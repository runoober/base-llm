from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from ..config import ExperimentConfig
from ..model.lm import SeekerOmniLM
from ..train.checkpoint import load_checkpoint


@torch.no_grad()
def generate_text(
    *,
    config_path: str | Path,
    ckpt_path: str | Path,
    tokenizer_dir: str | Path,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "auto",
) -> str:
    cfg = ExperimentConfig.load(config_path)
    tok = Tokenizer.from_file(str(Path(tokenizer_dir) / "tokenizer.json"))

    dev = torch.device("cuda" if (device == "auto" and torch.cuda.is_available()) else device)
    model = SeekerOmniLM(cfg.model).to(dev)
    load_checkpoint(Path(ckpt_path), model=model, optimizer=None)
    model.eval()

    input_ids = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=dev)
    out_ids = model.generate_text(
        input_ids,
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    text = tok.decode(out_ids[0].tolist(), skip_special_tokens=False)
    return text

