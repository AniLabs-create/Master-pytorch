# PyTorch – Chapter 00: Fundamentals

## Topics to Be Covered

1. Introduction to tensors: scalar, vector, matrix, and dimensions
2. Creating tensors: zeros, ones, rand, randn, arange, empty
3. Random seeds: `torch.manual_seed()`
4. Tensor data types and conversions
5. Tensor attributes and devices: dtype, shape, ndim, device
6. Tensor operations: addition, subtraction, multiplication and division
7. Matrix multiplication and dot products
8. Finding minimum, maximum, mean, sum and positional indices
9. Reshaping, stacking, squeezing and unsqueezing tensors
10. Permuting tensor dimensions
11. Indexing and selecting tensor elements
12. PyTorch tensors and NumPy arrays
13. Reproducibility and GPU/device considerations in practice

---

# Introduction

PyTorch is a Python framework that helps us build, train and use neural networks.

## What is a Tensor?

A tensor is the fundamental data structures we'll use throughout PyTorch.

---

## Setting Up PyTorch

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it in Windows PowerShell:

   ```powershell
   .\.venv/Scripts/Activate
   ```

3. Installing PyTorch:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install torch torchvision torchaudio
   ```

---

## Basic Tools in Tensor

**Example:**

```python
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
```

---

## Let's Compare All Six Functions

| Function | Purpose |
|---|---|
| `torch.zeros()` | Creates zeros |
| `torch.ones()` | Creates ones |
| `torch.rand()` | Uniform random values |
| `torch.randn()` | Normal random values |
| `torch.arange()` | Creates a sequence |
| `torch.empty()` | Uninitialized tensor |
| `torch.numel()` | Number of elements in the tensor |

Let's understand each one.

### Tensor Attributes

| Attribute | Purpose |
|---|---|
| `.shape` | Tells us the tensor's shape |
| `.ndim` | Tells us its number of dimensions |
| `.dtype` | Tells us its data type |
| `.device` | Tells us where it is stored |

---

# Lesson 4: Tensor Operations

## 1. Introduction

PyTorch allows us to perform mathematical operations on tensors. These operations are fundamental to neural networks and deep learning.

## 2. Basic Tensor Operations

```python
import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a + b)  # Addition: tensor([5, 7, 9])
print(a - b)  # Subtraction: tensor([-3, -3, -3])
print(a * b)  # Multiplication: tensor([4, 10, 18])
print(a / b)  # Division: tensor([0.25, 0.4, 0.5])
```

## 3. Scalar Operations

We can perform operations between a tensor and a single number.

```python
print(a + 10)  # tensor([11, 12, 13])
print(a * 2)   # tensor([2, 4, 6])
print(a ** 2)  # tensor([1, 4, 9])
```

PyTorch applies the operation to every element. This behavior is called **broadcasting**.

## 4. Element-wise Operations

When two tensors have compatible shapes, operations such as `+`, `-`, `*`, and `/` operate on corresponding elements.

**Example:**

```python
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(a * b)
```

**Output:**

```text
tensor([4, 10, 18])
```

## 5. Important Difference

- `*` → Element-wise multiplication.
- `@` → Matrix multiplication.
- `torch.matmul(a, b)` → Matrix multiplication.

We will study matrix multiplication in the next lesson.

## 6. Random Tensor Example

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

## 7. Key Takeaways

- PyTorch supports addition, subtraction, multiplication, and division.
- `*` performs element-wise multiplication, not matrix multiplication.
- Scalar operations are applied to every tensor element through broadcasting.
- `torch.manual_seed()` helps reproduce random tensor values.
- Tensor shapes must be compatible for mathematical operations.

# Lesson 5: Tensor Statistics

## 1. Introduction

PyTorch provides functions to calculate statistics from tensors. These functions help us find minimum values, maximum values, averages, sums and the positions of specific elements.

## 2. Finding the Minimum – `torch.min()`

`torch.min()` is used to find the smallest value inside a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([45, 78, 92, 34, 88])

result = torch.min(numbers)

print("Minimum value:", result)
```

**Output:**

```text
tensor(34)
```

## 3. Finding the Maximum – `torch.max()`

`torch.max()` is used to find the largest value inside a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([45, 78, 92, 34, 88])

result = torch.max(numbers)

print("Maximum value:", result)
```

**Output:**

```text
tensor(92)
```

## 4. Finding the Mean – `torch.mean()`

`torch.mean()` is used to calculate the average value of all elements in a tensor.

**Formula:**

Mean = Sum of all values / Number of values

**Example:**

```python
import torch

numbers = torch.tensor([45.0, 78.0, 92.0, 34.0, 88.0])

result = torch.mean(numbers)

print("Mean:", result)
```

**Output:**

```text
tensor(67.4000)
```

**Important:** `torch.mean()` requires a floating-point or complex tensor.

## 5. Finding the Sum – `torch.sum()`

`torch.sum()` is used to calculate the sum of all elements inside a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([10, 20, 30, 40, 50])

result = torch.sum(numbers)

print("Sum:", result)
```

**Output:**

```text
tensor(150)
```

## 6. Finding the Position of the Minimum – `torch.argmin()`

`torch.argmin()` returns the index (position) of the smallest element in a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([45, 78, 12, 34, 88])

minimum_value = torch.min(numbers)
minimum_index = torch.argmin(numbers)

print("Minimum value:", minimum_value)
print("Minimum index:", minimum_index)
```

**Output:**

```text
Minimum value: tensor(12)
Minimum index: tensor(2)
```

**Important:** Python and PyTorch use zero-based indexing. The first element has index `0`.

## 7. Finding the Position of the Maximum – `torch.argmax()`

`torch.argmax()` returns the index (position) of the largest element in a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([45, 78, 12, 34, 88])

maximum_value = torch.max(numbers)
maximum_index = torch.argmax(numbers)

print("Maximum value:", maximum_value)
print("Maximum index:", maximum_index)
```

**Output:**

```text
Maximum value: tensor(88)
Maximum index: tensor(4)
```

## 8. Comparison of All Six Functions

| Function | Purpose |
|---|---|
| `torch.min()` | Finds the smallest value |
| `torch.max()` | Finds the largest value |
| `torch.mean()` | Calculates the average |
| `torch.sum()` | Calculates the total |
| `torch.argmin()` | Finds the index of the smallest value |
| `torch.argmax()` | Finds the index of the largest value |

## 9. Complete Example

```python
import torch

number = torch.tensor([45.0, 78.0, 92.0, 34.0, 88.0])

min_value = torch.min(number)
max_value = torch.max(number)
mean_value = torch.mean(number)
sum_value = torch.sum(number)

min_index = torch.argmin(number)
max_index = torch.argmax(number)

print(f"Minimum Number: {min_value}")
print(f"Index of the minimum value is: {min_index}")

print(f"Maximum Number: {max_value}")
print(f"Index of the maximum value is: {max_index}")

print(f"The Average of all the given numbers is: {mean_value}")

print(f"The Sum of all the numbers in the list is: {sum_value}")
```

## 10. Key Takeaways

- `torch.min()` finds the smallest value.
- `torch.max()` finds the largest value.
- `torch.mean()` calculates the average.
- `torch.sum()` calculates the total.
- `torch.argmin()` returns the position of the smallest value.
- `torch.argmax()` returns the position of the largest value.
- PyTorch uses zero-based indexing.
- `torch.mean()` requires a floating-point or complex tensor.
- These functions return tensor results.



# Lesson 6: Reshaping, Stacking, Squeezing and Unsqueezing Tensors

## 1. `torch.reshape()`

`torch.reshape()` changes the shape of a tensor without changing its values.

**Example:**

```python
import torch

numbers = torch.tensor([1, 2, 3, 4, 5, 6])

result = torch.reshape(numbers, (2, 3))

print(result)
print(result.shape)
```

**Output:**
```text
tensor([[1, 2, 3],
        [4, 5, 6]])

torch.Size([2, 3])
```

**Important points:**
- Changes the shape of a tensor.
- Does not change the actual values.
- The total number of elements must remain the same.
- A tensor with 6 elements can be reshaped into `(2, 3)`, `(3, 2)`, or `(6,)`.

## 2. `torch.stack()`

`torch.stack()` combines multiple tensors by adding a new dimension.

**Example:**

```python
import torch

a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

b = torch.tensor([[2, 3, 1],
                  [1, 1, 3]])

result = torch.stack((a, b), dim=0)

print(result)
print(result.shape)
```

**Output:**
```text
tensor([[[1, 2, 3],
         [4, 5, 6]],

        [[2, 3, 1],
         [1, 1, 3]]])

torch.Size([2, 2, 3])
```

**Important points:**
- Combines tensors by creating a new dimension.
- All input tensors must have the same shape.
- `dim=0` adds the new dimension at the beginning.
- `dim=1` adds the new dimension at position 1.

**Syntax:**
```python
torch.stack((a, b), dim=0)
```

Remember to pass the tensors inside a tuple or list.

## 3. `torch.squeeze()`

`torch.squeeze()` removes dimensions of size `1` from a tensor.

**Example:**

```python
import torch

numbers = torch.tensor([[10, 20, 30]])

print(numbers.shape)

result = torch.squeeze(numbers)

print(result)
print(result.shape)
```

**Output:**
```text
torch.Size([1, 3])

tensor([10, 20, 30])

torch.Size([3])
```

**Important points:**
- Removes dimensions whose size is `1`.
- Does not change the actual tensor values.
- If no dimension has size `1`, the shape remains unchanged.
- `torch.squeeze(numbers, dim=0)` removes dimension 0 only if its size is 1.

**Example:**

A tensor with shape `(2, 3)` will remain `(2, 3)` after squeezing because neither dimension has size 1.

## 4. `torch.unsqueeze()`

`torch.unsqueeze()` adds a new dimension of size `1` to a tensor.

It is the opposite of `torch.squeeze()`.

**Example:**

```python
import torch

numbers = torch.tensor([10, 20, 30])

print("Original shape:", numbers.shape)

result = torch.unsqueeze(numbers, dim=0)

print(result)
print("New shape:", result.shape)
```

**Output:**
```text
Original shape: torch.Size([3])

tensor([[10, 20, 30]])

New shape: torch.Size([1, 3])
```

### Understanding the `dim` parameter

```python
numbers = torch.tensor([10, 20, 30])

result1 = torch.unsqueeze(numbers, dim=0)
result2 = torch.unsqueeze(numbers, dim=1)

print(result1.shape)  # torch.Size([1, 3])
print(result2.shape)  # torch.Size([3, 1])
```

- `dim=0`: Adds a dimension at the beginning.
- `dim=1`: Adds a dimension at position 1.

**Important points:**
- Adds a new dimension of size `1`.
- Does not change the actual tensor values.
- The position of the new dimension depends on `dim`.

## 5. Quick Comparison

| Function | Purpose |
|---|---|
| `torch.reshape()` | Changes the shape of a tensor |
| `torch.stack()` | Combines tensors by adding a new dimension |
| `torch.squeeze()` | Removes dimensions of size 1 |
| `torch.unsqueeze()` | Adds a dimension of size 1 |

## 6. Key Takeaways

1. `reshape()` changes the arrangement of a tensor's dimensions without changing its values.
2. `stack()` combines tensors and creates a new dimension.
3. `squeeze()` removes dimensions of size 1.
4. `unsqueeze()` adds dimensions of size 1.
5. These operations are useful when preparing tensors for neural networks and other tensor operations.

**Lesson 6 completed: Reshaping, Stacking, Squeezing and Unsqueezing Tensors.**