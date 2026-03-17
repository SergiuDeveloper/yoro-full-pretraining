import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = 'Qwen/Qwen2.5-0.5B'
CACHE_DIR = './cache'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16

MAX_NEW_TOKENS = 50
PROMPT = "How is"

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, torch_dtype=DTYPE).to(DEVICE)
model.eval()

print(f"Params: {sum(p.numel() for p in model.parameters()):,}\n")
print(f"Prompt: {PROMPT}\n")

input_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)

print("Generating...")
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)

print(f"\nModel output:\n{tokenizer.decode(output[0].tolist(), skip_special_tokens=True)}")
