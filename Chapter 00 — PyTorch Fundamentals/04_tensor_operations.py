import torch
torch.manual_seed(42)
first_mat = torch.rand(size=(3,3))
torch.manual_seed(100)
second_mat = torch.rand(size=(3,3))

print(f"Addition: {first_mat + second_mat}")
print(f"Second: {first_mat - second_mat}")
print(f"Multiplication: {first_mat*second_mat}")
print(f"Divisio: {first_mat / second_mat}")

# Create two matrices
A = torch.tensor([[1, 2],
                  [3, 4]])

B = torch.tensor([[5, 6],
                  [7, 8]])

# Element-wise multiplication
print("Element-wise multiplication:")
print(A * B)

# Matrix multiplication
print("Matrix multiplication:")
print(A @ B)

# Using torch.matmul()
print("Using torch.matmul():")
print(torch.matmul(A, B))

# Dot product
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

print("Dot product:")
print(torch.dot(x, y))