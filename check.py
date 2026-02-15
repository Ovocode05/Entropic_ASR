import torch
import bitsandbytes as bnb

print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0)}")
# If this fails, your environment setup is 'wrapper-like'—fix it now!
