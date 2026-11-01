import torch
torch.manual_seed(42)
first_mat = torch.rand(size=(3,3))
torch.manual_seed(100)
second_mat = torch.rand(size=(3,3))

print(f"Addition: {first_mat + second_mat}")
print(f"Second: {first_mat - second_mat}")
print(f"Multiplication: {first_mat*second_mat}")
print(f"Divisio: {first_mat / second_mat}")