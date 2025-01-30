import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
b = np.array([[2, 3, 2], [4, 5, 4], [6, 7, 6]])
print(b)

c = a @ b
print(c)


print(f"{a.shape} @ {b.shape} => {c.shape}")