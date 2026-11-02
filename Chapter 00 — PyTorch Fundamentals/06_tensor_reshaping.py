import torch
numbers = torch.tensor([1, 2, 3, 4, 5, 6])

print(f"Original Tensor: {numbers}")
print(f"Original Shape of the Tensor: {numbers.shape}")

changed = torch.reshape(numbers, (2,3))
print(f"Reshaped Tensor: {changed}")
print(f"Reshaped Shape of the tensor: {changed.shape}")