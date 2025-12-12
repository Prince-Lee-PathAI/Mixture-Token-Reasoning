import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from coconut.coconut import Coconut_no_shared

device = "cuda"

model_id = "gpt2"
base_model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_tokens("<|start-latent|>")
tokenizer.add_tokens("<|end-latent|>")
tokenizer.add_tokens("<|latent|>")

latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
start_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
end_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

base_model.resize_token_embeddings(len(tokenizer))

model =  Coconut_no_shared(
    base_model,
    latent_token_id=latent_id,
    start_latent_id=start_id,
    end_latent_id=end_id,
    eos_token_id=tokenizer.eos_token_id,
    state_combination_method="cross_attention",
    combination_use_gating=True,
).to(device)

state = torch.load("/workspace/checkpoint_best", map_location=device)
new_state = {k[7:] if k.startswith("module.") else k: v for k,v in state.items()}
model.load_state_dict(new_state)
model.eval()

torch.save(model.state_dict(), "/workspace/coconut_hf/pytorch_model.bin") # change this to your path