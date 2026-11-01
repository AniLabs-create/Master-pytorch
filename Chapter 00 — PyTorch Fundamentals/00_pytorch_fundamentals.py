import torch

my_tensor = torch.tensor([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

print("Tensor:")
print(my_tensor)

print("Shape:", my_tensor.shape)
print("Dimensions:", my_tensor.ndim)
print("Data type:", my_tensor.dtype)
print("Total elements:", my_tensor.numel())