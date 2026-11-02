import torch

numbers = torch.tensor([10, 20, 30])

print("Original tensor:")
print(numbers)

print("Original shape:", numbers.shape)

result = torch.unsqueeze(numbers, dim=0)

print("After unsqueezing:")
print(result)

print("New shape:", result.shape)