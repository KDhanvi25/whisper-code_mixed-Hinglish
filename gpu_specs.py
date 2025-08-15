import torch
print("PyTorch CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0))
print("Total memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)
