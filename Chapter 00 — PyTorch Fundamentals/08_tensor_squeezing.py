import torch
numbers = torch.tensor([[1,2,3]])
print(f"Original Tensor: {numbers}")
print(f"Shape of Original Tensor: {numbers.shape}")

changed = torch.squeeze(numbers)
print(f"Changed Tensor: {changed}")
print(f"Shape of Changed Tensor: {changed.shape}")