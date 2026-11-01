<===================Topics to be coverd====================>
1) Introduction to tensors: scalar, vector, matrix, and dimensions
2) Creating tensors: zeros, ones, rand, randn, arange, empty
3) Random seeds: torch.manual_seed()
4) Tensor data types and conversions
5) Tensor attributes and devices: dtype, shape, ndim, device
6) Tensor operations: addition, subtraction, multiplication and division
7) Matrix multiplication and dot products
8) Finding minimum, maximum, mean, sum and positional indices
9) Reshaping, stacking, squeezing and unsqueezing tensors
10) Permuting tensor dimensions
11) Indexing and selecting tensor elements
12) PyTorch tensors and NumPy arrays
13) Reproducibility and GPU/device considerations in practice



                        <=================Start from here========================>


Pytorch is a python framework that helps us build, train and use neural networks.

What is tensor:-
A tensor is the fundamental data structures we'll use throuhout pytorch.


setting up pytorch:-
1) Create a virtual envirnment
    python -m venv .venv
2) Activate it in windows powershell:-
    .\.venv/Scripts/Activate
3) Installing PyTorch:- 
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio

Basic tools in tensor :-
example:-
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


--->>

 Let's compare all six functions
Function	Purpose


torch.zeros()	Creates zeros
torch.ones()	Creates ones
torch.rand()	Uniform random values
torch.randn()	Normal random values
torch.arange()	Creates a sequence
torch.empty()	Uninitialized tensor
torch.numel()   Number of elements in the tensor


Let's understand each one.
Attribute	Purpose
.shape	Tells us the tensor's shape
.ndim	Tells us its number of dimensions
.dtype	Tells us its data type
.device	Tells us where it is stored



## Lesson 4: Tensor Operations

### 1. Introduction
PyTorch allows us to perform mathematical operations on tensors. These operations are fundamental to neural networks and deep learning.

### 2. Basic Tensor Operations

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)  # Addition: tensor([5, 7, 9])
print(a - b)  # Subtraction: tensor([-3, -3, -3])
print(a * b)  # Multiplication: tensor([4, 10, 18])
print(a / b)  # Division: tensor([0.25, 0.4, 0.5])
```

### 3. Scalar Operations

We can perform operations between a tensor and a single number.

```python
print(a + 10)  # tensor([11, 12, 13])
print(a * 2)   # tensor([2, 4, 6])
print(a ** 2)  # tensor([1, 4, 9])
```

PyTorch applies the operation to every element. This behavior is called **broadcasting**.

### 4. Element-wise Operations

When two tensors have compatible shapes, operations such as `+`, `-`, `*`, and `/` operate on corresponding elements.

Example:

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a * b)
```

Output:

```text
tensor([4, 10, 18])
```

### 5. Important Difference

- `*` → Element-wise multiplication.
- `@` → Matrix multiplication.
- `torch.matmul(a, b)` → Matrix multiplication.

We will study matrix multiplication in the next lesson.

### 6. Random Tensor Example

```python
torch.manual_seed(42)
first_mat = torch.rand(size=(3, 3))

torch.manual_seed(100)
second_mat = torch.rand(size=(3, 3))

print(first_mat + second_mat)
print(first_mat - second_mat)
print(first_mat * second_mat)
print(first_mat / second_mat)
```

### 7. Key Takeaways

- PyTorch supports addition, subtraction, multiplication, and division.
- `*` performs element-wise multiplication, not matrix multiplication.
- Scalar operations are applied to every tensor element through broadcasting.
- `torch.manual_seed()` helps reproduce random tensor values.
- Tensor shapes must be compatible for mathematical operations.