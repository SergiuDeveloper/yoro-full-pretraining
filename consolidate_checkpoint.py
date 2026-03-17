"""
Convert a DeepSpeed ZeRO checkpoint into a single best_model.pt.

Usage:
    modal run consolidate_checkpoint.py --checkpoint-tag step_73000 --output best_model_73000.pt
"""

import modal

HOURS = 60 * 60

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.0", "deepspeed", extra_options="--extra-index-url https://download.pytorch.org/whl/cu121")
)

app = modal.App("yoro-consolidate", image=image)

checkpoints_vol = modal.Volume.from_name("yoro-checkpoints", create_if_missing=False)


@app.function(
    cpu=4,
    memory=32768,
    volumes={"/checkpoints": checkpoints_vol},
    timeout=1 * HOURS,
)
def consolidate(checkpoint_tag: str, output: str):
    import torch
    import json
    from pathlib import Path
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

    checkpoint_dir = Path("/checkpoints") / checkpoint_tag
    output_path    = Path("/checkpoints") / output

    if not checkpoint_dir.exists():
        print(f"Checkpoint dir not found: {checkpoint_dir}")
        print("Available:", sorted(p.name for p in Path("/checkpoints").iterdir()))
        return

    print(f"Loading from /checkpoints/{checkpoint_tag} ...", flush=True)
    state_dict = get_fp32_state_dict_from_zero_checkpoint("/checkpoints", tag=checkpoint_tag)
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    print(f"{len(state_dict)} tensors loaded", flush=True)

    # Recover config from client state
    config = {}
    for f in checkpoint_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
            if "config" in data:
                config = data["config"]
                print(f"Config recovered from {f.name}")
                break
        except Exception:
            pass

    torch.save({
        "model_state_dict": state_dict,
        "config": config,
        "epoch": 0,
        "loss": 0.0,
    }, output_path)

    size_gb = output_path.stat().st_size / 1e9
    print(f"Saved to {output_path}  ({size_gb:.2f} GB)", flush=True)
    checkpoints_vol.commit()


@app.local_entrypoint()
def main(checkpoint_tag: str, output: str = "best_model.pt"):
    consolidate.remote(checkpoint_tag, output)
