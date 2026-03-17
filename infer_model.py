import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import flash_attn # noqa
except ImportError:
    # Flash attention not installed — patch from_pretrained to fall back to sdpa.
    _orig = AutoModelForCausalLM.from_pretrained.__func__
    @classmethod
    def _patched(cls, *args, **kwargs):
        if kwargs.get('attn_implementation') == 'flash_attention_2':
            kwargs['attn_implementation'] = 'sdpa'
        return _orig(cls, *args, **kwargs)
    AutoModelForCausalLM.from_pretrained = _patched

from subnet_model import SubnetLLM

CHECKPOINT_PATH = './best_model.pt'
CACHE_DIR = './cache'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16

MAX_NEW_TOKENS = 50
PROMPT = "How is"

print(f"Device: {DEVICE}")
print(f"Loading checkpoint: {CHECKPOINT_PATH}\n")

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
config = checkpoint['config']

MODEL_NAME = config.get('model_name', 'Qwen/Qwen2.5-0.5B')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Initializing SubnetLLM from checkpoint config...")
model = SubnetLLM(
    model_name=MODEL_NAME,
    cache_dir=CACHE_DIR,
    embedding_layers=config.get('embedding_layers', 4),
    coherence_layers=config.get('coherence_layers', 4),
    adaptation_layers=config.get('adaptation_layers', 4),
    compensation_layers=config.get('compensation_layers', 4),
    concatenation_layers=config.get('concatenation_layers', 4),
    device=DEVICE,
    dtype=DTYPE,
    freeze_base_model=config.get('freeze_base_model', True)
)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded from epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f}\n")

print(f"Prompt: {PROMPT}\n")

input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
attention_mask = torch.ones_like(input_ids)
cached_reasoning_outputs = [None]
generated = input_ids

print("Generating...")
with torch.no_grad():
    for i in range(MAX_NEW_TOKENS):
        logits, cached_reasoning_outputs = model(
            input_ids=generated,
            cached_reasoning_outputs=cached_reasoning_outputs,
            attention_mask=attention_mask,
            use_teacher_forcing=False
        )
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.ones_like(generated)

        if next_token.item() in (tokenizer.eos_token_id, tokenizer.pad_token_id):
            print(f"(stopped at EOS after {i + 1} tokens)")
            break

output_text = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)
print(f"\nModel output:\n{output_text}")
