import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import save_file
import json

def save_sharded_safetensors(state_dict, save_dir, prefix="model", max_shard_size=3 * 1024**3):
    """
    Save model weights as multiple sharded safetensors files with an index file.
    max_shard_size in bytes (default 5GB per shard)
    """
    os.makedirs(save_dir, exist_ok=True)

    shards = []
    shard_data = {}
    shard_size = 0
    shard_id = 0

    # Keeps map of parameter name to which shard and slices
    metadata = {}

    def save_current_shard():
        nonlocal shard_data, shard_id, shards
        shard_filename = f"{prefix}-{str(shard_id).zfill(5)}-of-{{num_shards}}.safetensors"
        shards.append(shard_filename)
        current_shard_path = os.path.join(save_dir, shard_filename)
        save_file(shard_data, current_shard_path)
        print(f"Saved shard {shard_id}: {current_shard_path}")
        shard_id += 1
        shard_data = {}
        return current_shard_path

    param_items = list(state_dict.items())
    total_params = len(param_items)

    for idx, (name, tensor) in enumerate(param_items):
        tensor_bytes = tensor.nbytes
        # If current shard size plus this tensor exceeds max, save current shard and start new one
        if shard_size + tensor_bytes > max_shard_size and shard_data:
            save_current_shard()
            shard_size = 0

        shard_data[name] = tensor
        metadata[name] = [shard_id, 0, tensor.shape]
        shard_size += tensor_bytes

    # Save the last shard
    save_current_shard()

    # Replace {num_shards} in shard filenames
    num_shards = len(shards)
    shards = [f"{prefix}-{str(i).zfill(5)}-of-{str(num_shards).zfill(5)}.safetensors" for i in range(num_shards)]

    # Rename shards to include total number
    for i in range(num_shards):
        old_path = os.path.join(save_dir, f"{prefix}-{str(i).zfill(5)}-of-{{num_shards}}.safetensors")
        new_path = os.path.join(save_dir, shards[i])
        os.rename(old_path, new_path)

    # Create index file
    index = {
        "metadata": {"total_size": sum(tensor.nbytes for tensor in state_dict.values())},
        "weight_map": {name: shards[shard_id] for name, (shard_id, _, _) in metadata.items()},
    }
    index_path = os.path.join(save_dir, f"{prefix}.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Saved index file: {index_path}")
    print(f"Created {num_shards} shards.")

def merge_adapter_with_base_model(base_model_name, adapter_name, tokenizer_name, save_dir):
    # Load extended tokenizer (with expanded vocab) first
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.float32)

    # Resize base model embeddings to tokenizer vocab size
    base_model.resize_token_embeddings(vocab_size)
    print(f"Resized base model embeddings to: {vocab_size}")

    # Load adapter on top of base model
    print("Loading adapter model...")
    adapter_model = PeftModel.from_pretrained(base_model, adapter_name)

    # Merge adapter weights into the base model
    print("Merging adapter weights into base model...")
    merged_model = adapter_model.merge_and_unload()

    # Save model weights in sharded safetensors format
    print("Saving merged model as sharded safetensors...")
    weights = {k: v.cpu() for k, v in merged_model.state_dict().items()}
    save_sharded_safetensors(weights, save_dir)

    # Save config and tokenizer files for Bedrock import
    print("Saving config and tokenizer...")
    merged_model.config.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print(f"Merged model saved with shards in directory: {save_dir}")

if __name__ == "__main__":
    base_model_name = "meta-llama/Meta-Llama-3-8B"
    adapter_name = "polyglots/SinLlama_v01"
    tokenizer_name = "polyglots/Extended-Sinhala-LLaMA"
    save_dir = "./merged_sinllama_8b"

    import os
    os.makedirs(save_dir, exist_ok=True)
    merge_adapter_with_base_model(base_model_name, adapter_name, tokenizer_name, save_dir)
