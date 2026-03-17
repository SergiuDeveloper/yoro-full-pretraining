import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import deepspeed
from transformers import AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset, Dataset
from datasets.distributed import split_dataset_by_node

from subnet_model import SubnetLLM

deepspeed.init_distributed()

local_rank = int(os.environ['LOCAL_RANK'])
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])

DEVICE = f'cuda:{local_rank}'
torch.cuda.set_device(local_rank)
torch.set_float32_matmul_precision('high')

IS_MAIN = rank == 0

def log(msg: str) -> None:
    if IS_MAIN:
        print(msg, flush=True)

MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
CACHE_DIR = './cache'
EMBEDDING_LAYERS = 4
COHERENCE_LAYERS = 4
ADAPTATION_LAYERS = 4
COMPENSATION_LAYERS = 4
CONCATENATION_LAYERS = 4

REINITIALIZE_WEIGHTS = True
FREEZE_BASE_MODEL = False

FINEWEB_DATASET = 'HuggingFaceFW/fineweb-edu'
FINEWEB_SUBSET = 'sample-10BT'
TARGET_TOKENS = 10_000_000_000
VAL_SEQUENCES = 10_000
CONTEXT_LENGTH = 512
SHUFFLE_BUFFER = 10_000

BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 1
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01

SANITY_CHECK_STEPS = 100_000
SANITY_CHECK_OUTPUT_TOKENS = 20
SANITY_PROMPT = 'Gravity is'

DTYPE = torch.bfloat16
LOSS_PLOT_PATH = 'loss.png'

SAVE_STEPS = 1_000
SAVE_DIR = Path(os.environ.get('CHECKPOINT_DIR', './checkpoints'))
CHECKPOINT_TAG = None  # e.g. 'step_5000' to resume; None = start fresh

DATA_DIR = os.environ.get('DATA_DIR')

DEEPSPEED_CONFIG = {
    'train_batch_size': BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size,
    'train_micro_batch_size_per_gpu': BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'optimizer': {
        'type': 'AdamW',
        'params': {
            'lr': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'betas': [0.9, 0.999],
            'eps': 1e-8
        }
    },
    'bf16': {'enabled': True},
    'zero_optimization': {
        'stage': 1,
        'reduce_bucket_size': 5e8,
        'overlap_comm': True,
    },
    'gradient_clipping': MAX_GRAD_NORM,
    'steps_per_print': 100,
    'wall_clock_breakdown': False
}

class FineWebIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        target_tokens: int,
        context_length: int,
        dataset_name: str,
        subset: str,
        shuffle_buffer: int,
        seed: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.tokenizer      = tokenizer
        self.target_tokens  = target_tokens
        self.context_length = context_length
        self.dataset_name   = dataset_name
        self.subset         = subset
        self.shuffle_buffer = shuffle_buffer
        self.seed           = seed
        self.rank           = rank
        self.world_size     = world_size

    def __iter__(self):
        if DATA_DIR:
            raw = load_dataset(self.dataset_name, name=self.subset, split='train', cache_dir=DATA_DIR)
            raw = raw.to_iterable_dataset(num_shards=128)
        else:
            raw = load_dataset(self.dataset_name, name=self.subset, split='train', streaming=True)
        raw = raw.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)
        raw = split_dataset_by_node(raw, rank=self.rank, world_size=self.world_size)

        buffer: List[int] = []
        token_count = 0
        eos = self.tokenizer.eos_token_id
        per_rank_tokens = self.target_tokens // self.world_size

        for example in raw:
            tokens = self.tokenizer.encode(example['text'], add_special_tokens=False)
            tokens.append(eos)
            buffer.extend(tokens)

            while len(buffer) >= self.context_length:
                chunk = buffer[:self.context_length]
                buffer = buffer[self.context_length:]
                yield {'input_ids': chunk}
                token_count += self.context_length
                if token_count >= per_rank_tokens:
                    return

def collect_val_dataset(
    tokenizer: AutoTokenizer,
    n_sequences: int,
    context_length: int,
    dataset_name: str,
    subset: str,
) -> Dataset:
    log(f'Collecting {n_sequences:,} validation sequences...')
    if DATA_DIR:
        raw = load_dataset(dataset_name, name=subset, split='train', cache_dir=DATA_DIR)
        raw = raw.to_iterable_dataset(num_shards=1)
    else:
        raw = load_dataset(dataset_name, name=subset, split='train', streaming=True)

    sequences = []
    buffer: List[int] = []
    eos = tokenizer.eos_token_id

    for example in raw:
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        tokens.append(eos)
        buffer.extend(tokens)

        while len(buffer) >= context_length and len(sequences) < n_sequences:
            sequences.append({'input_ids': buffer[:context_length]})
            buffer = buffer[context_length:]

        if len(sequences) >= n_sequences:
            break

    log(f'Val: {len(sequences):,} sequences ({len(sequences) * context_length:,} tokens)')
    return Dataset.from_list(sequences)

def collate_batch(batch, device: str) -> Dict[str, torch.Tensor]:
    input_ids = torch.tensor([s['input_ids'] for s in batch], dtype=torch.long)
    return {'input_ids': input_ids.to(device), 'labels': input_ids.to(device)}

def compute_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    del logits  # free before cross_entropy allocates its internal buffers
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

def save_training_checkpoint(
    engine: deepspeed.DeepSpeedEngine,
    epoch: int,
    global_step: int,
    train_losses: List[float],
    val_losses: List[float],
    best_val_loss: float,
    model_config: Dict,
) -> None:
    tag = f'step_{global_step}'
    engine.save_checkpoint(str(SAVE_DIR), tag=tag, client_state={
        'epoch': epoch,
        'global_step': global_step,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': model_config
    })
    if IS_MAIN:
        (SAVE_DIR / 'latest_tag.txt').write_text(tag)
        print(f'\nCheckpoint saved: {SAVE_DIR}/{tag}', flush=True)

def load_training_checkpoint(engine: deepspeed.DeepSpeedEngine, tag: str) -> Dict:
    log(f"Loading checkpoint '{tag}' from {SAVE_DIR}...")
    _, state = engine.load_checkpoint(str(SAVE_DIR), tag=tag)
    log(f"Resumed from epoch {state['epoch'] + 1}, step {state['global_step']}")
    return {
        'start_epoch': state['epoch'],
        'global_step': state['global_step'],
        'train_losses': state['train_losses'],
        'val_losses': state['val_losses'],
        'best_val_loss': state['best_val_loss']
    }

def train_epoch(
    engine: deepspeed.DeepSpeedEngine,
    dataset: torch.utils.data.IterableDataset,
    tokenizer: AutoTokenizer,
    epoch: int,
    global_step: int,
    train_losses: List[float],
    val_losses: List[float],
    best_val_loss: float,
    model_config: Dict,
) -> tuple[float, int]:
    engine.train()
    total_loss = 0.0
    num_batches = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=lambda batch: collate_batch(batch, 'cpu'),
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
    )

    # All ranks must process the same number of micro-batches or the ZeRO
    # allreduce collectives will hang waiting for ranks that have already
    # exhausted their shard. Cap every rank at this exact step count.
    max_steps = TARGET_TOKENS // (CONTEXT_LENGTH * BATCH_SIZE * world_size)
    max_steps = max(max_steps, 1)

    tokens_per_step = BATCH_SIZE * CONTEXT_LENGTH * world_size
    progress_bar = tqdm(
        total=TARGET_TOKENS,
        desc=f'Epoch {epoch + 1}',
        unit='tok', unit_scale=True,
        disable=not IS_MAIN,
        miniters=10_000 * tokens_per_step,
        mininterval=0,
    )

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break
        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
        labels    = batch['labels'].to(DEVICE, non_blocking=True)
        logits, _ = engine.module(
            input_ids=input_ids,
            use_teacher_forcing=True,
            prompt_length=128
        )
        loss = compute_lm_loss(logits, labels)
        engine.backward(loss)
        engine.step()

        if engine.is_gradient_accumulation_boundary():
            global_step += 1

            if global_step % SAVE_STEPS == 0:
                save_training_checkpoint(
                    engine, epoch, global_step,
                    train_losses, val_losses, best_val_loss, model_config
                )

            if IS_MAIN and global_step % SANITY_CHECK_STEPS == 0:
                engine.eval()
                with torch.no_grad():
                    prompt_ids = tokenizer.encode(SANITY_PROMPT, return_tensors='pt').to(DEVICE)
                    cached = [None]
                    generated = prompt_ids
                    for _ in range(SANITY_CHECK_OUTPUT_TOKENS):
                        out, cached = engine.module(
                            input_ids=generated,
                            cached_reasoning_outputs=cached,
                            attention_mask=torch.ones_like(generated),
                            use_teacher_forcing=False
                        )
                        next_tok = torch.argmax(out[:, -1, :], dim=-1, keepdim=True)
                        generated = torch.cat([generated, next_tok], dim=1)
                        if next_tok.item() == tokenizer.eos_token_id:
                            break
                    print('\nSanity:', tokenizer.decode(generated[0].tolist(), skip_special_tokens=True), flush=True)
                engine.train()

        total_loss += loss.item()
        num_batches += 1
        if IS_MAIN:
            progress_bar.update(tokens_per_step)
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'step': global_step})

    progress_bar.close()

    return (total_loss / num_batches if num_batches > 0 else 0.0, global_step)

def validate(engine: deepspeed.DeepSpeedEngine, dataset: Dataset) -> float:
    engine.eval()
    total_loss = 0.0
    num_batches = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, DEVICE),
        num_workers=0
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', disable=not IS_MAIN):
            logits, _ = engine.module(
                input_ids=batch['input_ids'],
                use_teacher_forcing=True,
                prompt_length=128
            )
            loss = compute_lm_loss(logits, batch['labels'])
            total_loss += loss.item()
            num_batches += 1

    loss_tensor = torch.tensor(total_loss / max(num_batches, 1), device=DEVICE)
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
    return loss_tensor.item()

def save_best_model(engine: deepspeed.DeepSpeedEngine, epoch: int, loss: float, model_config: Dict) -> None:
    if not IS_MAIN:
        return
    torch.save({
        'epoch': epoch,
        'model_state_dict': engine.module.state_dict(),
        'loss': loss,
        'config': model_config
    }, SAVE_DIR / 'best_model.pt')
    print(f'Best model saved to {SAVE_DIR / "best_model.pt"}', flush=True)

def plot_and_save_losses(train_losses: List[float], val_losses: List[float]) -> None:
    if not IS_MAIN:
        return
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f'Loss plot saved to {LOSS_PLOT_PATH}', flush=True)

if __name__ == '__main__':
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_dataset = collect_val_dataset(
        tokenizer, VAL_SEQUENCES, CONTEXT_LENGTH, FINEWEB_DATASET, FINEWEB_SUBSET
    )

    # Ensure all ranks finish dataset loading before any rank proceeds to model init.
    torch.distributed.barrier()
    torch.cuda.empty_cache()

    train_dataset = FineWebIterableDataset(
        tokenizer=tokenizer,
        target_tokens=TARGET_TOKENS,
        context_length=CONTEXT_LENGTH,
        dataset_name=FINEWEB_DATASET,
        subset=FINEWEB_SUBSET,
        shuffle_buffer=SHUFFLE_BUFFER,
        seed=42,
        rank=rank,
        world_size=world_size,
    )
    log(f'Train: streaming {TARGET_TOKENS:,} tokens across {world_size} GPUs  |  Val: {len(val_dataset):,} sequences')

    log('\nInitializing SubnetLLM...')
    model = SubnetLLM(
        model_name=MODEL_NAME,
        cache_dir=CACHE_DIR,
        embedding_layers=EMBEDDING_LAYERS,
        coherence_layers=COHERENCE_LAYERS,
        adaptation_layers=ADAPTATION_LAYERS,
        compensation_layers=COMPENSATION_LAYERS,
        concatenation_layers=CONCATENATION_LAYERS,
        device=DEVICE,
        dtype=DTYPE,
        freeze_base_model=FREEZE_BASE_MODEL
    )

    if REINITIALIZE_WEIGHTS:
        log('Reinitializing all weights...')
        def _reset(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(_reset)

    all_params       = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_count  = sum(p.numel() for p in trainable_params)
    log(f'\nTotal parameters:     {all_params:,}')
    log(f'Trainable parameters: {trainable_count:,}')
    log(f'Frozen parameters:    {all_params - trainable_count:,}')
    log(f'Training mode:        {"full" if not FREEZE_BASE_MODEL else "fine-tune (subnets only)"}')
    log(f'World size:           {world_size} GPUs')
    log(f'Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size}')

    free, total = torch.cuda.mem_get_info(local_rank)
    print(f'[rank{rank}] GPU memory: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total '
          f'({torch.cuda.memory_allocated(local_rank)/1e9:.1f} GB allocated)', flush=True)
    torch.distributed.barrier()
    torch.cuda.empty_cache()

    model_config = {
        'model_name': MODEL_NAME,
        'embedding_layers': EMBEDDING_LAYERS,
        'coherence_layers': COHERENCE_LAYERS,
        'adaptation_layers': ADAPTATION_LAYERS,
        'compensation_layers': COMPENSATION_LAYERS,
        'concatenation_layers': CONCATENATION_LAYERS,
        'reinitialize_weights': REINITIALIZE_WEIGHTS,
        'freeze_base_model': FREEZE_BASE_MODEL,
        'context_length': CONTEXT_LENGTH,
    }

    log('\nInitializing DeepSpeed engine...')
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=trainable_params,
        config=DEEPSPEED_CONFIG
    )
    log('DeepSpeed initialized.')

    start_epoch   = 0
    global_step   = 0
    best_val_loss = float('inf')
    train_losses: List[float] = []
    val_losses:   List[float] = []

    if CHECKPOINT_TAG is not None:
        state = load_training_checkpoint(engine, CHECKPOINT_TAG)
        start_epoch   = state['start_epoch']
        global_step   = state['global_step']
        best_val_loss = state['best_val_loss']
        train_losses  = state['train_losses']
        val_losses    = state['val_losses']
        log(f'Best val loss so far: {best_val_loss:.4f}')

    log('\nStarting training...')
    for epoch in range(start_epoch, NUM_EPOCHS):
        log(f'\nEpoch {epoch + 1}/{NUM_EPOCHS}')

        train_loss, global_step = train_epoch(
            engine=engine,
            dataset=train_dataset,
            tokenizer=tokenizer,
            epoch=epoch,
            global_step=global_step,
            train_losses=train_losses,
            val_losses=val_losses,
            best_val_loss=best_val_loss,
            model_config=model_config,
        )

        log(f'Training loss: {train_loss:.4f}')
        train_losses.append(train_loss)

        val_loss = validate(engine=engine, dataset=val_dataset)
        log(f'Validation loss: {val_loss:.4f}')
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_best_model(engine, epoch, val_loss, model_config)

    log('\nTraining completed!')
    plot_and_save_losses(train_losses, val_losses)
