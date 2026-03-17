"""
Modal pretraining entrypoint for YORO on a GPU cluster.

Setup (one-time):
    pip install modal
    modal setup                                      # authenticate
    modal secret create huggingface HF_TOKEN=hf_xxx  # HuggingFace token

Download the dataset to a persistent volume (run once before training):
    modal run pretrain_modal.py::prepare_data

Run training:
    modal run pretrain_modal.py::main

Checkpoints are stored in the 'yoro-checkpoints' Modal Volume and persist
across runs. Download them with:
    modal volume get yoro-checkpoints best_model.pt best_model.pt
"""

import modal
import os
import subprocess

HOURS = 60 * 60
GPUS_COUNT = 8
GPU_CONFIG = f"H100:{GPUS_COUNT}"  # e.g. "A100-40GB:1", "A100-40GB:8", "H100:8"

FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_SUBSET  = "sample-10BT"

# ---------------------------------------------------------------------------
# Container image — CUDA devel base so DeepSpeed can find CUDA_HOME
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.11"
    )
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "transformers>=4.44.0",
        "deepspeed",
        "datasets>=2.20.0",
        "huggingface_hub",
        "tqdm",
        "matplotlib",
        "mpi4py",
        "sentencepiece",
        "protobuf",
        extra_options="--extra-index-url https://download.pytorch.org/whl/cu121",
    )
    .pip_install("wheel", "packaging", "ninja")
    .pip_install(
        "flash-attn==2.6.3",
        extra_options="--no-build-isolation",
    )
    .add_local_file("subnet_model.py", "/root/subnet_model.py")
    .add_local_file("pretrain.py", "/root/pretrain.py")
)

app = modal.App("yoro-pretraining", image=image)

checkpoints_vol = modal.Volume.from_name("yoro-checkpoints", create_if_missing=True)
data_vol        = modal.Volume.from_name("yoro-data",        create_if_missing=True)

@app.function(
    cpu=4,
    memory=32768,
    volumes={"/data": data_vol},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=24 * HOURS,
)
def download_data():
    from datasets import load_dataset

    print(f"Downloading {FINEWEB_DATASET} ({FINEWEB_SUBSET}) to /data ...", flush=True)
    ds = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_SUBSET,
        split="train",
        cache_dir="/data",
        num_proc=4,
    )
    print(f"Downloaded {len(ds):,} examples.", flush=True)
    data_vol.commit()
    print("Volume committed.", flush=True)

@app.function(
    gpu=GPU_CONFIG,
    volumes={
        "/checkpoints": checkpoints_vol,
        "/data":        data_vol,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=6 * HOURS,
)
def train():
    os.chdir("/root")

    subprocess.run(
        [
            "deepspeed",
            "--num_gpus", str(GPUS_COUNT),
            "--master_addr", "localhost",
            "--master_port", "29500",
            "/root/pretrain.py",
        ],
        env={
            **os.environ,
            "CHECKPOINT_DIR":           "/checkpoints",
            "DATA_DIR":                 "/data",
            "PYTORCH_CUDA_ALLOC_CONF":  "expandable_segments:True",
        },
        check=True,
    )

    checkpoints_vol.commit()

@app.local_entrypoint()
def main():
    train.remote()

@app.local_entrypoint()
def prepare_data():
    download_data.remote()
