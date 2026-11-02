import torch

a = torch.tensor([[1,2,3],
                  [4,5,6]])
b = torch.tensor([[2,3,1],
                  [1,1,3]])
grouped = torch.stack((a, b), dim=1)
#dim=1 stacks the corresponding rows of the two tensors
#dim = 0 staks the tensors from the beginning
print(f"The tensor after stacking is is: {grouped}")

print(f"Shape of the new stacked tensor: {grouped.shape}")