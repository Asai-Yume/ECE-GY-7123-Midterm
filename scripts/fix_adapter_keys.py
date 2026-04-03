"""
Fix Unsloth Qwen3.5 adapter keys for text-only PEFT loading.

Unsloth saves:  base_model.model.model.language_model.layers.X...lora_A.weight
PEFT expects:   base_model.model.model.layers.X...lora_A.default.weight

Two remaps needed:
1. language_model.layers -> layers
2. lora_A.weight -> lora_A.default.weight (same for lora_B)

Usage:
    python fix_adapter_keys.py ./svg-lora-adapter-v5
    
Creates a fixed copy at ./svg-lora-adapter-v5_fixed/
"""

import os, sys, json, shutil
from safetensors import safe_open
from safetensors.torch import save_file

def fix_adapter(adapter_path):
    fixed_path = adapter_path.rstrip('/') + '_fixed'
    os.makedirs(fixed_path, exist_ok=True)
    
    # Copy all non-weight files
    for f in os.listdir(adapter_path):
        if not f.endswith('.safetensors') and not f.endswith('.bin'):
            shutil.copy2(os.path.join(adapter_path, f), os.path.join(fixed_path, f))
    
    # Fix adapter_config.json - change base model to text-only
    config_path = os.path.join(fixed_path, 'adapter_config.json')
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config['base_model_name_or_path'] = 'Qwen/Qwen3.5-2B'
        # Remove auto_mapping that references vision model
        config.pop('auto_mapping', None)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f'Fixed adapter_config.json')
    
    # Load and remap weights
    weights_file = os.path.join(adapter_path, 'adapter_model.safetensors')
    if not os.path.exists(weights_file):
        weights_file = os.path.join(adapter_path, 'adapter_model.bin')
    
    print(f'Loading weights from {weights_file}...')
    
    if weights_file.endswith('.safetensors'):
        old_weights = {}
        with safe_open(weights_file, framework='pt') as f:
            for key in f.keys():
                old_weights[key] = f.get_tensor(key)
    else:
        import torch
        old_weights = torch.load(weights_file, map_location='cpu')
    
    # Remap keys
    new_weights = {}
    remapped = 0
    for old_key, tensor in old_weights.items():
        new_key = old_key
        
        # Fix 1: language_model.layers -> layers
        new_key = new_key.replace('.language_model.layers.', '.layers.')
        
        # Fix 2: lora_A.weight -> lora_A.default.weight
        new_key = new_key.replace('.lora_A.weight', '.lora_A.default.weight')
        new_key = new_key.replace('.lora_B.weight', '.lora_B.default.weight')
        
        if new_key != old_key:
            remapped += 1
        new_weights[new_key] = tensor
    
    print(f'Remapped {remapped}/{len(old_weights)} keys')
    
    # Show sample mapping
    old_keys = sorted(old_weights.keys())
    new_keys = sorted(new_weights.keys())
    print(f'\nSample key mapping:')
    print(f'  OLD: {old_keys[0]}')
    print(f'  NEW: {new_keys[0]}')
    
    # Save fixed weights
    out_file = os.path.join(fixed_path, 'adapter_model.safetensors')
    save_file(new_weights, out_file)
    print(f'\nSaved fixed adapter to: {fixed_path}')
    print(f'Use this path for inference instead of the original.')
    return fixed_path

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fix_adapter_keys.py <adapter_path>')
        print('Example: python fix_adapter_keys.py ./svg-lora-adapter-v5')
        sys.exit(1)
    fix_adapter(sys.argv[1])
