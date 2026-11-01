import torch

# 1. Integer tensor
int_tensor = torch.tensor([10, 20, 30])
print(int_tensor)

# 2. Float tensor
float_tensor = torch.tensor([1.5, 2.5, 3.5])
print(float_tensor)

# 3. Convert integer tensor to float64
converted_tensor = int_tensor.to(torch.float64)
print(converted_tensor)
print(converted_tensor.dtype)

# 4. Random tensor
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)

# 5. Tensor attributes
print("Shape:", random_tensor.shape)
print("Dimensions:", random_tensor.ndim)
print("Data type:", random_tensor.dtype)
print("Device:", random_tensor.device)
print("CUDA available:", torch.cuda.is_available())
print("Total elements:", random_tensor.numel())